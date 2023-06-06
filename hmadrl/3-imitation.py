import argparse
import pathlib

import gym
import numpy as np
from imitation.algorithms import bc
from imitation.data.types import TrajectoryWithRew, TransitionsWithRew


def make_transitions(trajectories):
    transitions = []

    for trajectory_index in range(len(actions)):
        action, observation, reward, terminal = actions[trajectory_index], observations[trajectory_index], rewards[
            trajectory_index], terminals[trajectory_index]

        action = action[:-1]
        reward = reward[:-1]
        next_observation = observation[1:]
        observation = observation[:-1]
        done = np.full((len(observation),), False)
        done[-1] = terminal

        transitions.append(TransitionsWithRew(acts=action,
                                              obs=observation,
                                              rews=reward,
                                              dones=done,
                                              next_obs=next_observation,
                                              infos=np.empty((len(observation),))))

    return transitions


parser = argparse.ArgumentParser(description='Learn humanoid agent. Third step of HMADRL algorithm.')
parser.add_argument('--trajectory', type=pathlib.Path, help='path to file to trahectory from second step')
parser.add_argument('--human-model', type=pathlib.Path, help='path to file to save humanoid agent model')
args = parser.parse_args()

trajectories = np.load(args.trajectory)
actions = trajectories['actions']
observations = trajectories['observations']
rewards = trajectories['rewards']
terminals = trajectories['terminal']

trajectories = []

for trajectory_index in range(len(actions)):
    action, observation, reward, terminal = actions[trajectory_index], observations[trajectory_index], rewards[trajectory_index], terminals[trajectory_index]
    action = action[:-1]
    reward = reward[:-1]

    trajectories.append(TrajectoryWithRew(acts=action,
                                          obs=observation,
                                          rews=reward,
                                          terminal=terminal,
                                          infos=None))
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
bc_trainer.train(n_epochs=10)
bc_trainer.save_policy(args.human_model)