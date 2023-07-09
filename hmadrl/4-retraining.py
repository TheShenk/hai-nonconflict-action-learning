import json
import pathlib

from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import POlICY_REGISTRY, recursive_dict_update
from ray.rllib.policy.policy import PolicySpec

from hmadrl.custom_policy import ImitationPolicy
from hmadrl.imitation_registry import IMITATION_REGISTRY, RL_REGISTRY
from hmadrl.marllib_utils import find_checkpoint, create_policy_mapping, get_cc_config, find_latest_dir
from hmadrl.settings_utils import load_settings
from multiagent.env.ray_football import create_ma_football

ENV_REGISTRY["myfootball"] = create_ma_football
COOP_ENV_REGISTRY["myfootball"] = create_ma_football

parser = argparse.ArgumentParser(
    description='Retrain learned agents to play with human. Fourth step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                    help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

checkpoint_path = find_checkpoint(settings['multiagent']['algo']['name'],
                                  settings['env']['map'],
                                  settings['multiagent']['model']['core_arch'],
                                  settings['save']['multiagent_model'])
model_path = checkpoint_path
params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'
with open(params_path, 'r') as params_file:
    multiagent_params = json.load(params_file)

experiment_path = find_latest_dir(pathlib.Path(settings['save']['human_model']), lambda obj: obj.is_dir())
humanoid_model_path = str(experiment_path / 'model.zip')

inner_algo_cls = RL_REGISTRY[settings['imitation']['inner_algo']['name']]
humanoid_model = IMITATION_REGISTRY[settings['imitation']['algo']['name']].load(humanoid_model_path, 'cpu',
                                                                                inner_algo_cls)

env = marl.make_env(environment_name=settings['env']['name'],
                    map_name=settings['env']['map'],
                    **settings['env']['args'])
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                          **multiagent_params['model']['custom_model_config']['algo_args'])
model = marl.build_model(env, algo, multiagent_params['model']['custom_model_config']['model_arch_args'])

env_instance, env_info = env
model_class, model_info = model

policies = {f'policy_{agent_num}': PolicySpec() for agent_num, agent_id in enumerate(env_instance.agents) if agent_id != settings["rollout"]["human_agent"]}
policies["human"] = PolicySpec(ImitationPolicy(humanoid_model, model_class))

policy_mapping = create_policy_mapping(env_instance)
policy_mapping[settings["rollout"]["human_agent"]] = "human"


def policy_mapping_fn(agent_id):
    return policy_mapping[agent_id]


exp_info = env_info
exp_info = recursive_dict_update(exp_info, model_info)
exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

exp_info['algorithm'] = settings['multiagent']['algo']['name']
exp_info['restore_path'] = {
    "params_path": params_path,
    "model_path": model_path
}
exp_info["stop_timesteps"] = settings['retraining']['timesteps']
exp_info['local_dir'] = settings['save']['retraining_model']
exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies, policy_mapping_fn)

algo_runner = POlICY_REGISTRY[settings['multiagent']['algo']['name']]
result = algo_runner(model_class, exp_info, run_config, env_info, stop_config, None)
