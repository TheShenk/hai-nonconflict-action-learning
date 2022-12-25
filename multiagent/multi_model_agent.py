from typing import List, Optional, Tuple, Type

import numpy as np
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from tqdm import tqdm

from agents.base_agent import BaseAgent
from multiagent.action_combiners import NON_VEC_COMBINER
from multiagent.multi_agent_proxy import MultiAgentProxy


class MultiModelAgent(BaseAgent):

    def __init__(self,
                 env: GymEnv,
                 models: List[MultiAgentProxy],
                 static_models: Optional[List[BaseAgent]] = None,
                 actions_combiner=NON_VEC_COMBINER):
        super().__init__(env)
        self.models = models
        self.static_models = static_models if static_models is not None else []
        self.actions_combiner = actions_combiner

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
        return self.actions_combiner(actions)

    def learn(self,
              total_timesteps: int,
              callback: MaybeCallback = None):

        observation, reward, done, info = self.env.reset(), 0, np.ones((self.env.num_envs,)), None
        total_reward = 0

        time = 0

        for model in self.models:
            model.start_learning(total_timesteps)
        callback.on_training_start(locals(), globals())

        with tqdm(total=total_timesteps) as pbar:
            while time < total_timesteps:
                current_step_reward = 0

                for model in self.models:
                    model.start_record()

                callback.on_rollout_start()

                while any([model.continue_record() for model in self.models]):
                    sample_actions_results = [model.sample_action() for model in self.models] + \
                                             [s_model.predict(observation, episode_start=done) for s_model in
                                              self.static_models]
                    actions = tuple(map(lambda x: x[0], sample_actions_results))
                    total_action = self.actions_combiner(actions)

                    next_observation, reward, done, info = self.env.step(total_action)
                    total_reward += reward
                    current_step_reward += reward

                    time += self.env.num_envs
                    pbar.update(self.env.num_envs)
                    callback.on_step()

                    for model, action, sample_actions_result in zip(self.models, actions, sample_actions_results):
                        model.record(observation, action, next_observation, reward, done, info, sample_actions_result)

                    observation = next_observation

                callback.on_rollout_end()

                for model in self.models:
                    model.end_record()
                    model.train()

        callback.on_training_end()
