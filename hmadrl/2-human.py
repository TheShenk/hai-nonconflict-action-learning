import numpy as np
import pygame
from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY

from hmadrl.custom_policy import PyGamePolicy
from hmadrl.human_recorder import HumanRecorder
from hmadrl.marllib_utils import load_trainer
from hmadrl.presetted_agents_env import PreSettedAgentsEnv

from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca


parser = argparse.ArgumentParser(description='Collect human trajectories. Second step of HMADRL algorithm.')
parser.add_argument('--env', default='myfootball', type=str, help='name of environment (default: myfootball)')
parser.add_argument('--map', default='hca', type=str, help='name of map (default: hca)')
parser.add_argument('--algo', default='mappo', type=str, help='name of learning algorithm (default: mappo)')
parser.add_argument('--episodes', default=15, type=int, help='number of episodes (default: 5)')
parser.add_argument('--checkpoint', type=str, help='path to checkpoint from first step')
parser.add_argument('--trajectory', type=str, help='path to file to save human actions and observations')

args = parser.parse_args()


def human_policy(key, obs):
    enemy_goal_position = np.array([1, 0])
    ball_position = obs[:2]
    ball_velocity = obs[2:4]

    ball_enemy_goal_vector = enemy_goal_position - ball_position
    ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

    move_direction = [0, 0]
    hit_direction = ball_enemy_goal_vector / ball_enemy_goal_distance

    if key == pygame.K_RIGHT or key == pygame.K_d:
        move_direction = [1, 0]
    elif key == pygame.K_DOWN or key == pygame.K_s:
        move_direction = [0, 1]
    elif key == pygame.K_LEFT or key == pygame.K_a:
        move_direction = [-1, 0]
    elif key == pygame.K_UP or key == pygame.K_w:
        move_direction = [0, -1]
    elif key == pygame.K_SPACE:
        hit_direction = [1, 0]

    return np.append(move_direction, hit_direction)


env = marl.make_env(environment_name=args.env, map_name=args.map)
env_instance, _ = env
trainer = load_trainer(args.algo, env, args.checkpoint)


def rollout(env, policy, episodes_count):
    for episode_index in range(episodes_count):
        done = False
        observation = env.reset()
        while not done:
            action = policy(observation)
            observation, reward, done, info = env.step(action)
            rollout_env.render('human')
    env.close()


rollout_env = PreSettedAgentsEnv(HumanRecorder(env_instance, 'player_0', args.trajectory),
                                 {'player_1': trainer.get_policy('policy_1')}, 'player_0')
rollout_policy = PyGamePolicy(human_policy)
rollout(rollout_env, rollout_policy, args.episodes)