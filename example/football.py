import numpy as np
import pygame

from agents.simple_football_agents import AsPolicy, attacking, goalkeeper
from hmadrl.pygame_utils import PyGamePolicy
from hmadrl.marllib_utils import register_env

from hmadrl.presetted_agents_env import PresetAgents
from multiagent.env.football import TwoSideFootball
from multiagent.env.multiagent_football import MultiAgentFootball


def create_football(env_config):
    if env_config['map_name'] == 'hca-discrete':
        env = TwoSideFootball(number_of_player=2, action_space_type=["discrete", "box"])
        env = MultiAgentFootball(env)
        env = PresetAgents(env, {"blue_0": AsPolicy(attacking, 0, 4, [1, 0]),
                                 "blue_1": AsPolicy(goalkeeper, 1, 4, [-0.8, 0], [1, 0])})
        return env
    if env_config['map_name'] == 'hca':
        env = TwoSideFootball(number_of_player=2, action_space_type=["box", "box"])
        env = MultiAgentFootball(env)
        env = PresetAgents(env, {"blue_0": AsPolicy(attacking, 0, 4, [1, 0]),
                                 "blue_1": AsPolicy(goalkeeper, 1, 4, [-0.8, 0], [1, 0])})
        return env
    if env_config['map_name'] == 'full':
        env = TwoSideFootball(number_of_player=2, action_space_type=["box", "box"])
        env = MultiAgentFootball(env)
        return env


register_env("myfootball", create_football, 300,
             {"all_scenario": {
                    "description": "both commands intelligent",
                    "team_prefix": ("red_", "blue_"),
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": True
             }})


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


policy = PyGamePolicy(human_policy)
