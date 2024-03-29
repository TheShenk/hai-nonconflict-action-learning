from typing import List, Optional, Tuple, Type, Callable

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from tqdm import tqdm

import utils
from agents.base_agent import BaseAgent
from multiagent.action_combiners import NON_VEC_COMBINER, BOX_COMBINER
from multiagent.callbacks import MACallback
from multiagent.multi_agent_proxy import MultiAgentProxy


class MultiModelAgent(BaseAgent):

    def __init__(self,
                 env: GymEnv,
                 models: Optional[List[MultiAgentProxy]] = None,
                 static_models: Optional[List[BaseAgent]] = None,
                 nvec_actions_combiner = NON_VEC_COMBINER,
                 vec_actions_combiner = BOX_COMBINER):
        """

        :param env: Среда, в которой действует агент.
        :param models: Список моделей которые способны обучаться.
        :param static_models: Список статичных моделей. Их действия добавляются к действиям обычных моделей, но не
        используются для обучения.
        :param nvec_actions_combiner: Функция для объединения действий, когда известны действия ото всех агентов для
        одной среды.
        :param vec_actions_combiner: Функция для обединения действий, когда известны действия всех агентов во всех
        средах и они представлены как [[агент1 в среде1, ..., агент1 в средеM],
                                                         ...,
                                       [агентN в среде1, ..., агентN в средеM]
         ].
        """
        super().__init__(env)

        self.models = models if models is not None else []
        self.static_models = static_models if static_models is not None else []
        self.nvec_actions_combiner = nvec_actions_combiner
        self.vec_actions_combiner = vec_actions_combiner
        self.time = 0

    def save(self, path):
        for index, model in enumerate(self.models):
            model.save(f"{path}-{index}")

    def load(self, proxy_cls: Type[MultiAgentProxy], model_cls, models_count: int, *args, **kwargs):
        self.models = []
        for index in range(models_count):
            model = proxy_cls.load(model_cls, *args, **kwargs)
            self.models.append(model)
        return self

    def _predict(self,
                 observation: np.ndarray,
                 state: Optional[Tuple[np.ndarray, ...]] = None,
                 episode_start: Optional[np.ndarray] = None,
                 deterministic: bool = False):

        actions = np.array([model.predict(observation, state, episode_start, deterministic)[0]
                            for model in self.models+self.static_models])
        return self.nvec_actions_combiner(actions)

    def learn(self,
              total_timesteps: int,
              callback: Optional[MACallback] = None):

        if callback is None:
            callback = MACallback()

        total_models_count = len(self.models) + len(self.static_models)
        observations, dones = [self.env.reset(),] * total_models_count, np.ones((self.env.num_envs,))

        self.time = 0

        for model in self.models:
            model.start_learning(total_timesteps)

        callback.init_callback(self)
        callback.on_training_start()

        with tqdm(total=total_timesteps) as pbar:
            while self.time < total_timesteps:

                for model in self.models:
                    model.start_record()

                callback.on_rollout_start()

                while any([model.continue_record() for model in self.models]):
                    next_observations, rewards, dones, infos, sample_actions_results \
                        = self.collect_step_info(observations, dones)

                    self.time += self.env.num_envs
                    pbar.update(self.env.num_envs)
                    callback.on_step()

                    for model, obs, n_obs, rew, sar in zip(self.models,
                                                           observations, next_observations, rewards,
                                                           sample_actions_results):
                        model.record(obs, n_obs, rew, dones, infos, sar)

                    observations = next_observations

                callback.on_rollout_end()

                for model in self.models:
                    model.end_record()
                    model.train()

        callback.on_training_end()

    def collect_step_info(self, observations, done):

        models_count = len(self.models)
        static_models_count = len(self.static_models)
        total_models_count = models_count+static_models_count

        sample_actions_results = []
        for model_index in range(models_count):
            sample_actions_results.append(self.models[model_index].sample_action())
        for s_model_index in range(static_models_count):
            model_observation = observations[models_count+s_model_index]
            sample_actions_results.append(self.static_models[s_model_index].predict(model_observation, episode_start=done))

        actions = tuple(map(lambda x: x[0], sample_actions_results))
        total_action = self.vec_actions_combiner(actions)

        next_observation, reward, done, info = self.env.step(total_action)

        return [next_observation,] * total_models_count, [reward,] * total_models_count, done, info, sample_actions_results
