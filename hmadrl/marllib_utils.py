import json
import os.path
import pathlib

import cloudpickle
from marllib.marl import recursive_dict_update, dict_update, _Algo
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
from marllib.marl.algos.core.VD.iql_vdn_qmix import JointQTrainer
from marllib.marl.algos.core.VD.vda2c import VDA2CTrainer
from marllib.marl.algos.core.VD.vdppo import VDPPOTrainer
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.util.ml_utils.dict import merge_dicts
from typing import Dict, Tuple, Any, Callable


def get_cc_config(exp_info, env, stop, policies, policy_mapping_fn):
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
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_1"]
        },
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
    render_config = {
        "evaluation_interval": 1,
        "evaluation_num_episodes": 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "record_env": True,
            "render_env": True,
        }
    }

    run_config = recursive_dict_update(run_config, render_config)

    render_stop_config = {
        "training_iteration": 1,
    }

    stop_config = recursive_dict_update(stop_config, render_stop_config)

    return exp_info, run_config, env_info, stop_config, restore_config


def find_latest_dir(dir: pathlib.Path, filter_fn: Callable[[pathlib.Path], bool]) -> pathlib.Path:
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
    checkpoint = [item for item in checkpoint_dir.iterdir() if not item.name.startswith('.') and not item.suffixes]
    assert len(checkpoint) == 1, checkpoint
    return str(checkpoint[0])


def get_trainer_class(algo_name, config):
    TRAINER_REGISTER = {
        'ippo': IPPOTrainer,
        'mappo': MAPPOTrainer,
        'vdppo': VDPPOTrainer,
        'happo': HAPPOTrainer(config), # FIXME: step 4

        'itrpo': TRPOTrainer,
        'matrpo': MATRPOTrainer,
        'hatrpo': HATRPOTrainer, # FIXME: step 4

        'ia2c': IA2CTrainer,
        'maa2c': MAA2CTrainer,
        'coma': COMATrainer, # Use discrete action space
        'vda2c': VDA2CTrainer,

        'iddpg': IDDPGTrainer,
        'maddpg': MADDPGTrainer, # FIXME: step 4
        'facmac': FACMACTrainer, # FIXME: step 4

        # 'iql': JointQTrainer,  # Don't support individual learning
        # 'vdn': JointQTrainer,  # Don't support individual learning
        # 'qmix': JointQTrainer  # Don't support individual learning
    }

    return TRAINER_REGISTER[algo_name]


def load_trainer(algo: _Algo, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], local_dir_path: str):
    env_instance, env_info = env
    model_class, model_info = model

    checkpoint_path = find_checkpoint(algo.name,
                                      env_info['env_args']['map_name'],
                                      model_info['model_arch_args']['core_arch'],
                                      local_dir_path)

    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint = cloudpickle.load(checkpoint_file)

    params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'

    with open(params_path, 'r') as params_file:
        params = json.load(params_file)

    worker = cloudpickle.loads(checkpoint['worker'])
    policies: Dict[str, PolicySpec] = worker['policy_specs']

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
    params["model"]["custom_model_config"]["space_obs"] = env_instance.observation_space
    params["model"]["custom_model_config"]["space_act"] = env_instance.action_space
    if algo.name in {'iddpg', 'maddpg', 'facmac'}:
        ModelCatalog.register_custom_model("DDPG_Model", model_class)
    else:
        ModelCatalog.register_custom_model("current_model", model_class)
        params["model"]["custom_model"] = "current_model"

    trainer_cls = get_trainer_class(algo.name, params)
    trainer = trainer_cls(params)
    trainer.restore(checkpoint_path)
    return trainer


def create_policy_mapping(env: MultiAgentEnv) -> Dict[str, str]:
    policy_mapping = {agent_id: f"policy_{agent_number}" for agent_number, agent_id in enumerate(env.agents)}
    return policy_mapping
