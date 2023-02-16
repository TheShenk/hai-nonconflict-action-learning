from typing import List, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from agents.base_agent import BaseAgent


class MultiEnvAgent(BaseAgent):

    def __init__(self, env: VecEnv, models: List[Union[BaseAlgorithm, BaseAgent]]):
        super().__init__(env)
        self.models = models

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):
        pass

    def predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):

        actions = []
        for env_index in range(self.env.num_envs):
            current_obs = observation[env_index]
            current_ep_start = episode_start[env_index] if episode_start else None
            current_action = self.models.predict(current_obs, state, current_ep_start, deterministic)
            actions.append(current_action)

        return actions, None

