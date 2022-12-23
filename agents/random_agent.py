from typing import Optional, Tuple

import gym
import numpy as np

from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env: gym.Env, action_space=None):
        super().__init__(env)
        self.env = env
        self.action_space = action_space

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        if self.action_space:
            return self.action_space.sample(), None
        return self.env.action_space.sample(), None