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

        marl_env = isinstance(self.env.observation_space, list) or hasattr(self.env, "observation_spaces")

        # Проверка на то, является ли текущая среда векторной. Это не может быть isinstanceof(..., VecEnv), так как
        # при обучении используется evaluate_policy, которая не заменяет текущую среду. Обычная нейронная сеть
        # в таком случае полагается на политики, которые это как-то учитывают.
        # TODO: Вроде, MAEvaluateCallback теперь всегда использует DummyVecEnv. Нужно проверить и, скорее всего,
        #  заменить эту проверку
        vec_env = not marl_env and observation.ndim > len(self.env.observation_space.shape)

        if vec_env:
            actions = []
            for i in range(observation.shape[0]):
                action = self._predict(observation[i], episode_start=episode_start)
                actions.append(action)
            return np.reshape(actions, (-1,) + self.env.action_space.shape), None
        else:
            action = self._predict(observation, state, episode_start, deterministic)
            return action, None