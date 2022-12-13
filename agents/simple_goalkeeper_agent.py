import gym
from agents.base_agent import BaseAgent


class SimpleGoalkeeperAgent(BaseAgent):

    def __init__(self, env: gym.Env,
                 player_index: int,
                 player_obs_len: int = 4):
        super().__init__(env=env)
        self.player_index = player_index
        self.player_obs_len = player_obs_len
        self.goal_position = [self.env.width]

    def predict(self, observation):
        ball_position = observation[:2]
        ball_velocity = observation[2:4]
        players_observation = observation[4:]
        goalkeeper_obs_start = self.player_index * self.player_obs_len
        goalkeeper_position = players_observation[goalkeeper_obs_start:goalkeeper_obs_start + 2]
        goalkeeper_velocity = players_observation[goalkeeper_obs_start + 2:goalkeeper_obs_start + 4]


        print(ball_velocity, ball_position)
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4)).sample(), None
