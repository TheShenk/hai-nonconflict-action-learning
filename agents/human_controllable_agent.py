from typing import Optional, Tuple

import gym
import numpy as np
import pygame
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent


class HumanControllableAgent(BaseAgent):

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

        ball_enemy_goal_vector = self.enemy_goal_position - ball_position
        ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

        move_direction = [0, 0]
        hit_direction = ball_enemy_goal_vector/ball_enemy_goal_distance

        events = pygame.event.get()
        for event in events:

            if event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_RIGHT | pygame.K_d:
                        move_direction = [1, 0]
                    case pygame.K_DOWN | pygame.K_s:
                        move_direction = [0, 1]
                    case pygame.K_LEFT | pygame.K_a:
                        move_direction = [-1, 0]
                    case pygame.K_UP | pygame.K_w:
                        move_direction = [0, -1]
                    case pygame.K_SPACE:
                        hit_direction = [1, 0]

        return [np.append(move_direction, hit_direction)]





