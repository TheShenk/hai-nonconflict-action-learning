import argparse
import pathlib

import gym
import numpy as np
from imitation.algorithms import bc
from imitation.data.rollout import TrajectoryAccumulator
from imitation.data.types import TrajectoryWithRew, TransitionsWithRew


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
bc_trainer = bc.BC(
    observation_space=gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float64
        ),
    action_space=gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),
    demonstrations=trajectories,
    rng=rng,
    device='cpu'
)
bc_trainer.train(n_epochs=1000)
bc_trainer.save_policy(args.human_model)