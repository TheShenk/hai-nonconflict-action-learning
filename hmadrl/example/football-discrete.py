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
    if env_config['map_name'] == 'full-discrete':
        env = TwoSideFootball(number_of_player=2, action_space_type=["discrete", "discrete"])
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

    direction = 0
    action = 1
    if key == pygame.K_UP or key == pygame.K_w:
        direction = 3
    elif key == pygame.K_RIGHT or key == pygame.K_d:
        direction = 2
    elif key == pygame.K_DOWN or key == pygame.K_s:
        direction = 1
    elif key == pygame.K_LEFT or key == pygame.K_a:
        direction = 4
    elif key == pygame.K_SPACE:
        action = 2

    return direction * 5 + action


policy = PyGamePolicy(human_policy)
