from typing import Optional, Tuple

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent


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

        ball_position = observation[:2]
        ball_velocity = observation[2:4]

        players_observation = observation[4:]
        attacking_obs_start = self.player_index * self.player_obs_len
        attacking_position = players_observation[attacking_obs_start:attacking_obs_start + 2]
        attacking_velocity = players_observation[attacking_obs_start + 2:attacking_obs_start + 4]

        to_ball_vector = ball_position - attacking_position
        to_ball_distance = np.linalg.norm(to_ball_vector)

        ball_enemy_goal_vector = self.enemy_goal_position - ball_position
        ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

        return [np.append(np.append(
            to_ball_vector / to_ball_distance,
            ball_enemy_goal_vector / ball_enemy_goal_distance
        ), [0, ] * self.message_dims_number)]


