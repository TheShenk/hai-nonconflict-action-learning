import time
from itertools import chain

import gym.spaces
import numpy as np
import torch
import wandb

from offpolicy.utils.util import is_multidiscrete
from offpolicy.runner.mlp.base_runner import MlpRunner

from agents.random_agent import RandomAgent
from gym_futbol.envs_v1 import Futbol
from gym_futbol.envs_v1.futbol_env import TOTAL_TIME, NUMBER_OF_PLAYER

class MARLBenchmarkFootball(Futbol):

    def __init__(self, total_time=TOTAL_TIME, debug=False,
                 number_of_player=NUMBER_OF_PLAYER, team_B_model=RandomAgent,
                 action_space_type="box", random_position=False,
                 team_reward_coeff=10, ball_reward_coeff=10, message_dims_number=0,
                 is_out_rule_enabled=True):
        super().__init__(total_time=total_time,
                         debug=debug,
                         number_of_player=number_of_player,
                         team_B_model=team_B_model,
                         action_space_type=action_space_type,
                         random_position=random_position,
                         team_reward_coeff=team_reward_coeff,
                         ball_reward_coeff=ball_reward_coeff,
                         message_dims_number=message_dims_number,
                         is_out_rule_enabled=is_out_rule_enabled)

        self.observation_space = [gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20,),
            dtype=np.float64
        ),] * number_of_player
        self.share_observation_space = [gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(20 * number_of_player,),
            dtype=np.float64
        ),] * number_of_player
        # self.observation_space = self.share_observation_space
        self.action_space = [gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)),] * number_of_player

    def reset(self):
         obs = super().reset()
         return [obs,] * self.number_of_player

    def step(self, action):
        obs, rew, done, info = super().step(action)
        return ([obs,] * self.number_of_player,
                [[rew,],] * self.number_of_player,
                [[done,],] * self.number_of_player,
                info)


