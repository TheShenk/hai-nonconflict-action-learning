import argparse
import pathlib

import numpy as np
from imitation.data.types import TrajectoryWithRew
from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.marllib_utils import load_trainer
from hmadrl.presetted_agents_env import PreSettedAgentsEnv

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
parser.add_argument('--env', default='myfootball', type=str, help='name of environment (default: myfootball)')
parser.add_argument('--map', default='hca', type=str, help='name of map (default: hca)')
parser.add_argument('--algo', default='mappo', type=str, help='name of learning algorithm (default: mappo)')
parser.add_argument('--imit-algo', default='airl', type=str, help='name of imitation algorithm (default: airl)')
parser.add_argument('--base-algo', default='ppo', type=str, help='name of base imitation algorithm (default: ppo)')
parser.add_argument('--checkpoint', type=str, help='path to checkpoint from first step')
parser.add_argument('--trajectory', type=pathlib.Path, help='path to file to trahectory from second step')
parser.add_argument('--human-model', type=pathlib.Path, help='path to file to save humanoid agent model')
args = parser.parse_args()

trajectories = np.load(args.trajectory)
actions = trajectories['actions']
observations = trajectories['observations']
rewards = trajectories['rewards']
dones = trajectories['dones']

trajectories = make_trajectories(actions, observations, rewards, dones)
rng = np.random.default_rng(0)

env = marl.make_env(environment_name=args.env, map_name=args.map)
env_instance, _ = env
trainer = load_trainer(args.algo, env, args.checkpoint)

rollout_env = PreSettedAgentsEnv(env_instance, {'player_1': trainer.get_policy('policy_1')}, 'player_0')
rollout_env = make_vec_env(lambda: rollout_env, n_envs=1)

trainer = IMITATION_REGISTRY[args.imit_algo](rollout_env, trajectories, rng, PPO)
trainer.train(100000)
trainer.save(args.human_model)
