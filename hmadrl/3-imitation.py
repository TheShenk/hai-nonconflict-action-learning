import argparse

import numpy as np
from imitation.data.types import TrajectoryWithRew
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from stable_baselines3.common.env_util import make_vec_env

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.marllib_utils import load_trainer, create_policy_mapping
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, create_inner_algo_from_settings

from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca


def make_trajectories(actions, observations, rewards, dones: np.ndarray):
    trajectories = []
    done_indexes, = np.where(dones)
    observation_shift = 0

    for previous_done, current_done in zip(np.append([0], done_indexes[:-1]), done_indexes):

        trajectories.append(TrajectoryWithRew(acts=actions[previous_done:current_done],
                                              obs=observations[previous_done + observation_shift:current_done + observation_shift + 1],
                                              rews=rewards[previous_done:current_done],
                                              infos=np.empty((current_done-previous_done,)),
                                              terminal=True))
        observation_shift += 1

    return trajectories


parser = argparse.ArgumentParser(description='Learn humanoid agent. Third step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, nargs=1, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

trajectories = np.load(settings['save']['trajectory'])
actions = trajectories['actions']
observations = trajectories['observations']
rewards = trajectories['rewards']
dones = trajectories['dones']

trajectories = make_trajectories(actions, observations, rewards, dones)
rng = np.random.default_rng(0)

env = marl.make_env(environment_name=settings['env']['name'],
                    map_name=settings['env']['map'],
                    **settings['env']['args'])
env_instance, _ = env
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                          **settings['multiagent']['algo']['args'])
model = marl.build_model(env, algo, settings['multiagent']['model'])
trainer = load_trainer(algo, env, model, settings['save']['multiagent_model'])

policy_mapping = create_policy_mapping(env_instance)
policy_mapping = {agent_id: trainer.get_policy(policy_id) for agent_id, policy_id in policy_mapping.items()}
human_agent = settings['rollout']['human_agent']
policy_mapping.pop(human_agent, None)

rollout_env = PreSettedAgentsEnv(env_instance, policy_mapping, human_agent)
rollout_env = make_vec_env(lambda: rollout_env, n_envs=1)

inner_algo = create_inner_algo_from_settings(rollout_env, settings)
trainer = IMITATION_REGISTRY[settings['imitation']['algo']['name']](rollout_env, trajectories, rng, inner_algo, settings['imitation']['algo']['args'])
trainer.train(settings['imitation']['timesteps'])
trainer.save(settings['save']['human_model'])
