from typing import Optional, Tuple

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent
class SimpleGoalkeeperAgent(BaseAgent):

    def __init__(self, env: GymEnv, player_index: int, player_obs_len: int = 4, message_dims_number: int = 0):
        super().__init__(env)
        self.player_index = player_index
        self.player_obs_len = player_obs_len
        self.goal_position = np.array([-0.8, 0])
        self.enemy_goal_position = np.array([1, 0])
        self.message_dims_number = message_dims_number

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):

        ball_position = observation[:2]
        ball_velocity = observation[2:4]
        future_ball_position = ball_position + ball_velocity * 0.5

        players_observation = observation[4:]
        goalkeeper_obs_start = self.player_index * self.player_obs_len
        goalkeeper_position = players_observation[goalkeeper_obs_start:goalkeeper_obs_start + 2]
        goalkeeper_velocity = players_observation[goalkeeper_obs_start + 2:goalkeeper_obs_start + 4]
        future_goalkeeper_position = goalkeeper_position + goalkeeper_velocity

        to_goal_vector = self.goal_position - goalkeeper_position
        to_goal_distance = np.linalg.norm(to_goal_vector)

        to_ball_vector = future_ball_position - goalkeeper_position
        to_ball_distance = np.linalg.norm(to_ball_vector)

        ball_enemy_goal_vector = self.enemy_goal_position - ball_position
        ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

        if to_ball_distance < 0.45:
            return [np.append(np.append(
                to_ball_vector / to_ball_distance,
                ball_enemy_goal_vector / ball_enemy_goal_distance
            ), [1, ] * self.message_dims_number)]
        else:
            return [np.append(to_goal_vector / to_goal_distance, [[0, 0] + [1,] * self.message_dims_number])]
