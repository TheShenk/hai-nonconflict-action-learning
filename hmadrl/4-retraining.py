import pathlib

from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY
from marllib.marl import POlICY_REGISTRY, recursive_dict_update
from ray.rllib.policy.policy import PolicySpec

from hmadrl.custom_policy import ImitationPolicy
from hmadrl.imitation_registry import IMITATION_REGISTRY, RL_REGISTRY
from hmadrl.marllib_utils import find_checkpoint, create_policy_mapping, get_cc_config
from hmadrl.presetted_agents_env import PreSettedAgentsMultiEnv
from hmadrl.settings_utils import load_settings
from multiagent.env.ray_football import create_ma_football_hca

ENV_REGISTRY["myfootball"] = create_ma_football_hca
COOP_ENV_REGISTRY["myfootball"] = create_ma_football_hca

parser = argparse.ArgumentParser(
    description='Retrain learned agents to play with human. Fourth step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, nargs=1,
                    help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

inner_algo_cls = RL_REGISTRY[settings['imitation']['inner_algo']['name']]
humanoid_model = IMITATION_REGISTRY[settings['imitation']['algo']['name']].load(settings['save']['human_model'], 'cpu',
                                                                                inner_algo_cls)

# create_env_fn = ENV_REGISTRY[settings['env']['name']]
# ENV_REGISTRY[settings['env']['name']] = lambda config: PreSettedAgentsMultiEnv(create_env_fn(config),
#                                                                                {settings['rollout'][
#                                                                                     'human_agent']: humanoid_model})
# coop_create_env_fn = COOP_ENV_REGISTRY[settings['env']['name']]
# COOP_ENV_REGISTRY[settings['env']['name']] = lambda config: PreSettedAgentsMultiEnv(coop_create_env_fn(config),
#                                                                                     {settings['rollout'][
#                                                                                          'human_agent']: humanoid_model})

env = marl.make_env(environment_name=settings['env']['name'],
                    map_name=settings['env']['map'],
                    **settings['env']['args'])
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                          **settings['multiagent']['algo']['args'])
model = marl.build_model(env, algo, settings['multiagent']['model'])

env_instance, env_info = env
model_class, model_info = model

checkpoint_path = find_checkpoint(algo.name,
                                  env_info['env_args']['map_name'],
                                  model_info['model_arch_args']['core_arch'],
                                  settings['save']['multiagent_model'])

model_path = checkpoint_path
params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'

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
exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies, policy_mapping_fn)

algo_runner = POlICY_REGISTRY[settings['multiagent']['algo']['name']]
result = algo_runner(model_class, exp_info, run_config, env_info, stop_config, None)

# algo.fit(env, model,
#          share_policy='individual',
#          stop={'timesteps_total': settings['multiagent']['timesteps']},
#          local_dir=settings['save']['retraining_model'],
#          restore_path={
#              'params_path': params_path,
#              'model_path': model_path
#          })
