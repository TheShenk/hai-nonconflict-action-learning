from typing import Optional, Tuple

import gym
import numpy as np
from stable_baselines3.common.type_aliases import GymEnv

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env: GymEnv, action_space=None):
        super().__init__(env)
        self.action_space = action_space

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        if self.action_space:
            return self.action_space.sample()
        return self.env.action_space.sample()