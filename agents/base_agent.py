from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv


class BaseAgent:

    def __init__(self, env: GymEnv):
        self.env = env
    @abstractmethod
    def predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        pass