from typing import Optional, Tuple

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv

from agents.base_agent import BaseAgent


class ConstantAgent(BaseAgent):
    def __init__(self, env: GymEnv, action):
        super().__init__(env)
        self.action = action

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        return self.action