import random
from typing import List, Union, Optional, Tuple

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.type_aliases import GymEnv

from agents.base_agent import BaseAgent


class RandomMultiModelAgent(BaseAgent):

    def __init__(self, env: GymEnv, models: List[Union[BaseAlgorithm, BaseAgent]]):
        super().__init__(env)
        self.models = models
        self.current_model = random.choice(models)

    def _predict(self,
                observation: np.ndarray,
                state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None,
                deterministic: bool = False):

        if episode_start.all():
            self.current_model = random.choice(self.models)
        return self.current_model.predict(observation, state, episode_start, deterministic)[0]