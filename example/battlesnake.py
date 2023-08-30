import numpy as np
from numpy.typing import NDArray

from gymnasium.spaces import Discrete
from gymnasium.spaces.utils import unflatten

import pygame
from pettingzoo.utils import parallel_to_aec, aec_to_parallel

from hagl import HAGLParallelWrapper
from hmadrl.MARLlibWrapper import TimeLimit
from hmadrl.pygame_utils import PyGamePolicy, PyGameFPSWrapper
from hmadrl.marllib_utils import register_env

from pypzbattlesnake.env import BattleSnake


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: NDArray[np.int64]) -> np.int64:
    nonzero = np.nonzero(x)
    if len(nonzero[0]) == 0:
        return space.start
    return space.start + nonzero[0][0]


def create_battlesnake(env_config):
    env = BattleSnake(2, 2)
    env = HAGLParallelWrapper(env)
    env = TimeLimit(env, 300)
    env = PyGameFPSWrapper(parallel_to_aec(env), fps=2)
    env = aec_to_parallel(env)
    return env


register_env("battlesnake", create_battlesnake, 300,
             {"all_scenario": {
                    "description": "both commands intelligent",
                    "team_prefix": ("snake_0", "snake_1"),
                    "all_agents_one_policy": True,
                    "one_agent_one_policy": True
             }})


# TODO: use HAGL
def human_policy(key, obs):

    action = [0, 0, 0, 0, 1]
    if key == pygame.K_UP or key == pygame.K_w:
        action = [1, 0, 0, 0, 0]
    elif key == pygame.K_DOWN or key == pygame.K_s:
        action = [0, 1, 0, 0, 0]
    elif key == pygame.K_LEFT or key == pygame.K_a:
        action = [0, 0, 1, 0, 0]
    elif key == pygame.K_RIGHT or key == pygame.K_d:
        action = [0, 0, 0, 1, 0]

    return np.array(action)


policy = PyGamePolicy(human_policy)
