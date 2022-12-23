from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv


class BaseAgent:

    def __init__(self, env: GymEnv):
        self.env = env
    @abstractmethod
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
        # Проверка на то, является ли текущая среда векторной. Это не может быть isinstanceof(..., VecEnv), так как
        # при обучении используется evaluate_policy, которая не заменяет текущую среду. Обычная нейронная сеть
        # в таком случае полагается на политики, которые это как-то учитывают.
        if observation.ndim > len(self.env.observation_space.shape):
            actions = []
            for i in range(observation.shape[0]):
                action = self._predict(observation[i])
                actions.append(action)
            return np.reshape(actions, (-1,) + self.env.action_space.shape), None
        else:
            action = self._predict(observation, state, episode_start, deterministic)
            return action, None