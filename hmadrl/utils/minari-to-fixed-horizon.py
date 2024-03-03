import shutil

import gymnasium
import argparse

import minari
import numpy as np
from minari.dataset.minari_dataset import parse_dataset_id

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--duration', type=int)
args = parser.parse_args()

fixed_episode_data = []
episode_rewards = []

current_episode_length = 0
current_observation = []
current_actions = []
current_rewards = []

dataset = minari.MinariDataset(args.file)
for episode in dataset.iterate_episodes():
    timesteps_count = min(args.duration - current_episode_length, episode.total_timesteps)
    if current_episode_length + timesteps_count == args.duration and timesteps_count == episode.total_timesteps:
        timesteps_count -= 1

    current_actions.append(episode.actions[:timesteps_count])
    current_rewards.append(episode.rewards[:timesteps_count])

    if current_episode_length + timesteps_count == args.duration:
        current_observation.append(episode.observations[:timesteps_count + 1])
        terminations = np.zeros((args.duration,))
        truncations = np.zeros((args.duration,))
        truncations[-1] = True
        fixed_episode_data.append({
            'observations': np.concatenate(current_observation, axis=0),
            'actions': np.concatenate(current_actions, axis=0),
            'rewards': np.concatenate(current_rewards, axis=0),
            'terminations': terminations,
            'truncations': truncations
        })
        episode_rewards.append(np.sum(fixed_episode_data[-1]['rewards']))

        current_episode_length = 0
        current_observation = []
        current_actions = []
        current_rewards = []
    else:
        current_observation.append(episode.observations[:-1])
        current_episode_length += timesteps_count

env = gymnasium.Env()
env.spec = dataset.spec.env_spec

env_name, dataset_name, version = parse_dataset_id(dataset.spec.dataset_id)
fixed_dataset_id = f"{env_name}-{dataset_name}_fixed-v{version}"
fixed_dataset = minari.create_dataset_from_buffers(
    fixed_dataset_id,
    env,
    fixed_episode_data,
    action_space=dataset.spec.action_space,
    observation_space=dataset.spec.observation_space,
    minari_version="~=0.4.1"
)
shutil.move(fixed_dataset.spec.data_path, args.output)
minari.delete_dataset(fixed_dataset_id)
print(f'Episodes: {fixed_dataset.total_episodes} ({fixed_dataset.total_steps} steps), '
      f'Reward: {np.mean(episode_rewards)} +- {np.std(episode_rewards)}')