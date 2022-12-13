import gym
import numpy as np

from agents.base_agent import BaseAgent
class SimpleGoalkeeperAgent(BaseAgent):

    def __init__(self, env: gym.Env,
                 player_index: int,
                 player_obs_len: int = 4):
        super().__init__(env=env)
        self.player_index = player_index
        self.player_obs_len = player_obs_len
        self.goal_position = np.array([-0.8, 0])
        self.enemy_goal_position = np.array([1, 0])

    def predict(self, observation):
        ball_position = observation[:2]
        ball_velocity = observation[2:4]

        players_observation = observation[4:]
        goalkeeper_obs_start = self.player_index * self.player_obs_len
        goalkeeper_position = players_observation[goalkeeper_obs_start:goalkeeper_obs_start + 2]
        goalkeeper_velocity = players_observation[goalkeeper_obs_start + 2:goalkeeper_obs_start + 4]

        to_goal_vector = self.goal_position - goalkeeper_position
        to_goal_distance = np.linalg.norm(to_goal_vector)

        to_ball_vector = ball_position - goalkeeper_position
        to_ball_distance = np.linalg.norm(to_ball_vector)

        ball_enemy_goal_vector = self.enemy_goal_position - ball_position
        ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

        if to_goal_distance < 0.2 and to_ball_distance < 0.2:
            return [np.append(to_ball_vector / to_ball_distance, ball_enemy_goal_vector / ball_enemy_goal_distance)], None
        else:
            return [np.append(to_goal_vector / to_goal_distance, [[0, 0]])], None
