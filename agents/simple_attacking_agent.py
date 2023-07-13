from typing import Optional, Tuple

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent
from agents.simple_football_agents import attacking


class SimpleAttackingAgent(BaseAgent):

    def __init__(self,  env: GymEnv, player_index: int, player_obs_len: int = 4, message_dims_number: int = 0):
        super().__init__(env)
        self.player_index = player_index
        self.player_obs_len = player_obs_len
        self.enemy_goal_position = np.array([1, 0])
        self.message_dims_number = message_dims_number

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):

        return [np.append(attacking(observation, self.player_index, self.player_obs_len, self.enemy_goal_position),
                [0, ] * self.message_dims_number)]


