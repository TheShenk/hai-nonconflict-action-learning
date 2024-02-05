import json
import os.path
import pathlib
from typing import Dict, Tuple, Any, Callable

import cloudpickle
import gym
import pettingzoo
import ray.tune
import supersuit
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import recursive_dict_update, dict_update, _Algo, set_ray
from marllib.marl.algos.core.CC.coma import COMATrainer
from marllib.marl.algos.core.CC.happo import HAPPOTrainer
from marllib.marl.algos.core.CC.hatrpo import HATRPOTrainer
from marllib.marl.algos.core.CC.maa2c import MAA2CTrainer
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from marllib.marl.algos.core.CC.matrpo import MATRPOTrainer
from marllib.marl.algos.core.CC.maddpg import MADDPGTrainer
from marllib.marl.algos.core.IL.a2c import IA2CTrainer
from marllib.marl.algos.core.IL.ddpg import IDDPGTrainer
from marllib.marl.algos.core.IL.ppo import IPPOTrainer
from marllib.marl.algos.core.IL.trpo import TRPOTrainer
from marllib.marl.algos.core.VD.facmac import FACMACTrainer
from marllib.marl.algos.core.VD.vda2c import VDA2CTrainer
from marllib.marl.algos.core.VD.vdppo import VDPPOTrainer

from hmadrl.MARLlibWrapper import MARLlibWrapper, CoopMARLlibWrapper, TimeLimit
from hmadrl.settings_utils import get_save_settings, find_checkpoint_in_dir


def get_config(exp_info, env, stop, multiagent_config):
    env_info = env.get_env_info()
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()

    run_config = {
        "seed": int(exp_info["seed"]),
        "env": exp_info["env"] + "_" + exp_info["env_args"]["map_name"],
        "num_gpus_per_worker": exp_info["num_gpus_per_worker"],
        "num_gpus": exp_info["num_gpus"],
        "num_workers": exp_info["num_workers"],
        "multiagent": multiagent_config,
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }
    stop_config = dict_update(stop_config, stop)
    restore_config = exp_info['restore_path']

    return exp_info, run_config, env_info, stop_config, restore_config


def find_latest_dir(dir: pathlib.Path, filter_fn: Callable[[pathlib.Path], bool] = lambda _: True) -> pathlib.Path:
    subdirs = [item for item in dir.iterdir() if filter_fn(item)]
    subdirs.sort(key=lambda subdir: os.path.getmtime(subdir))
    return subdirs[-1]


def find_checkpoint(algo_name: str, map_name: str, core_arch: str, local_dir_path: str):
    local_dir = pathlib.Path(local_dir_path)
    model_dir = find_latest_dir(local_dir,
                                lambda item: item.is_dir() and
                                             item.name.startswith(f"{algo_name}_{core_arch}_{map_name}"))
    experiment_dir = find_latest_dir(model_dir, lambda item: item.is_dir())
    checkpoint_dir = find_latest_dir(experiment_dir, lambda item: item.is_dir())
    return str(find_checkpoint_in_dir(checkpoint_dir))


def get_trainer_class(algo_name, config):
    TRAINER_REGISTER = {
        'ippo': IPPOTrainer,
        'mappo': MAPPOTrainer,
        'vdppo': VDPPOTrainer,
        'happo': HAPPOTrainer(config),

        'itrpo': TRPOTrainer,
        'matrpo': MATRPOTrainer,
        'hatrpo': HATRPOTrainer,

        'ia2c': IA2CTrainer,
        'maa2c': MAA2CTrainer,
        'coma': COMATrainer, # Use discrete action space
        'vda2c': VDA2CTrainer,

        'iddpg': IDDPGTrainer, # Can't be restored in ray 1.8.0: https://github.com/ray-project/ray/pull/22245
        # #TODO: Maybe use another noise?
        # 'maddpg': MADDPGTrainer, # MADDPG and FACMAC can't be run becouse of before_learn_on_batch in
        # marl/algos/utils/centralized_Q. It calls in maddpg.py and assume that all policies from backup is learning.
        # That wrong in case of HMADRL as at step 4 human policies don't learn.
        # 'facmac': FACMACTrainer,

        # 'iql': JointQTrainer,  # Don't support individual learning
        # 'vdn': JointQTrainer,  # Don't support individual learning
        # 'qmix': JointQTrainer  # Don't support individual learning
    }

    return TRAINER_REGISTER[algo_name]


def load_trainer_from_checkpoint(checkpoint_path, custom_model=None):

    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint = cloudpickle.load(checkpoint_file)

    params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'

    with open(params_path, 'r') as params_file:
        params = json.load(params_file)

    worker = cloudpickle.loads(checkpoint['worker'])
    policies: Dict[str, PolicySpec] = worker['policy_specs']

    policy_name = list(policies.keys())[0]
    observation_space = policies[policy_name].observation_space
    action_space = policies[policy_name].action_space

    recursive_dict_update(params,
                          {
                              "framework": "torch",
                              "multiagent": {
                                  "policy_mapping_fn": lambda: None,
                                  "policies_to_train": []
                              },
                              "num_workers": 1,
                              "num_gpus": 0,
                              "num_cpus_per_worker": 1,
                              "num_gpus_per_worker": 0,
                          })

    # This line could be in dict_update but standatr vd configs have policies of type string, not dict. Because of
    # this recursive_dict_update will throw exception.
    params["multiagent"]["policies"] = policies
    params["model"]["custom_model_config"]["space_obs"] = gym.spaces.Dict({"obs": observation_space})
    params["model"]["custom_model_config"]["space_act"] = action_space

    if custom_model is not None:
        params["model"]["custom_model"] = "current_model"

    params.pop("env")
    trainer_cls = get_trainer_class(params["model"]["custom_model_config"]["algorithm"], params)
    trainer = trainer_cls(params)
    trainer.restore(checkpoint_path)
    return trainer


def load_trainer(algo: _Algo, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], multiagent_save_settings):
    env_instance, env_info = env
    model_class, model_info = model

    local_dir, restore_path = get_save_settings(multiagent_save_settings)
    if not restore_path["model_path"]:
        checkpoint_path = find_checkpoint(algo.name,
                                          env_info['env_args']['map_name'],
                                          model_info['model_arch_args']['core_arch'],
                                          local_dir)
    else:
        checkpoint_path = restore_path["model_path"]

    custom_model = None
    if algo.name in {'iddpg', 'maddpg', 'facmac'}:
        ModelCatalog.register_custom_model("DDPG_Model", model_class)
    else:
        ModelCatalog.register_custom_model("current_model", model_class)
        custom_model = "current_model"

    return load_trainer_from_checkpoint(checkpoint_path, custom_model)


def create_policy_mapping(env: MultiAgentEnv) -> Dict[str, str]:
    policy_mapping = {agent_id: f"policy_{agent_number}" for agent_number, agent_id in enumerate(env.possible_agents)}
    return policy_mapping


def rollout(env, policy, episodes_count):
    episode_reward = []
    for episode_index in range(episodes_count):
        print("Current episode:", episode_index)
        done = False
        observation, _ = env.reset()
        current_episode_reward = 0
        while not done:
            action = policy(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            current_episode_reward += reward
            env.render()
        episode_reward.append(current_episode_reward)
    return sum(episode_reward)/episodes_count


def register_env(environment_name: str,
                 create_fn: Callable[[dict], pettingzoo.ParallelEnv],
                 max_episode_len: int,
                 policy_mapping_info: dict):

    def create_marllib_fn(config: dict):
        env = create_fn(config)
        return MARLlibWrapper(env, max_episode_len, policy_mapping_info)

    def create_coop_marllib_fn(config: dict):
        env = create_fn(config)
        return CoopMARLlibWrapper(env, max_episode_len, policy_mapping_info)

    ENV_REGISTRY[environment_name] = create_marllib_fn
    COOP_ENV_REGISTRY[environment_name] = create_coop_marllib_fn


def make_env(environment_settings: dict):
    marllib_env_config = environment_settings.copy()

    marllib_env_config["env"] = environment_settings["name"]
    marllib_env_config.pop("name")

    marllib_env_config["env_args"] = environment_settings["args"]
    marllib_env_config.pop("args")

    marllib_env_config["env_args"]["map_name"] = environment_settings["map"]
    marllib_env_config = set_ray(marllib_env_config)

    env_reg_name = marllib_env_config["env"] + "_" + marllib_env_config["env_args"]["map_name"]

    if marllib_env_config.get("force_coop", False):
        ray.tune.register_env(env_reg_name, lambda _: COOP_ENV_REGISTRY[marllib_env_config["env"]](marllib_env_config["env_args"]))
        env = COOP_ENV_REGISTRY[marllib_env_config["env"]](marllib_env_config["env_args"])
    else:
        ray.tune.register_env(env_reg_name, lambda _: ENV_REGISTRY[marllib_env_config["env"]](marllib_env_config["env_args"]))
        env = ENV_REGISTRY[marllib_env_config["env"]](marllib_env_config["env_args"])

    return env, marllib_env_config

STEP_NAME = "error"
