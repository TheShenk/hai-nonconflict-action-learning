from typing import List, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from agents.base_agent import BaseAgent
from multiagent.action_combiners import BOX_COMBINER, NON_VEC_COMBINER
from multiagent.multi_agent_proxy import MultiAgentProxy
from multiagent.multi_model_agent import MultiModelAgent


class TwoSideModelAgent(MultiModelAgent):

    def __init__(self,
                 env: VecEnv,
                 left_models: List[MultiAgentProxy],
                 right_models: List[MultiAgentProxy],
                 left_static_models: Optional[List[BaseAgent]] = None,
                 right_static_models: Optional[List[BaseAgent]] = None,
                 nvec_actions_combiner=NON_VEC_COMBINER,
                 vec_actions_combiner=BOX_COMBINER):

        self.left_static_models = left_static_models if left_static_models is not None else []
        self.right_static_models = right_static_models if right_static_models is not None else []

        models = left_models + right_models
        static_models = self.left_static_models + self.right_static_models
        super().__init__(env, models, static_models,
                         nvec_actions_combiner=nvec_actions_combiner,
                         vec_actions_combiner=vec_actions_combiner)

        self.left_models = left_models
        self.right_models = right_models

    # Переопределение _predict необходимо, чтобы сохранить поддержку evaluate_policy. Здесь возвращаются только
    # действия левой команды. При этом сохранение происходит для моделей всех сторон, так как при этом используется
    # save из MultiModelAgent, сохраняющий все из self.models.
    def _predict(self,
                 observation: np.ndarray,
                 state: Optional[Tuple[np.ndarray, ...]] = None,
                 episode_start: Optional[np.ndarray] = None,
                 deterministic: bool = False):
        actions = np.array([model.predict(observation, state, episode_start, deterministic)[0]
                            for model in self.left_models])
        return self.nvec_actions_combiner(actions)


    def get_side_total_action(self, side_models, side_static_models, observation, done):

        models_count = len(side_models)
        static_models_count = len(side_static_models)

        sample_actions_result = [model.sample_action() for model in side_models]
        for s_model_index in range(static_models_count):
            model_observation = observation[models_count + s_model_index]
            sample_actions_result.append(self.static_models[s_model_index].predict(model_observation, done))

        actions = tuple(map(lambda sar: sar[0], sample_actions_result))
        total_action = self.vec_actions_combiner(actions)
        # Возвращать необходимо только sar, используемые для реальных моделей, а не статических. Иначе в record
        # из learn попадет sar от статических моделей.
        return total_action, sample_actions_result[:models_count]

    def collect_step_info(self, observation, done):

        left_total_action, left_sample_actions_result = \
            self.get_side_total_action(self.left_models, self.left_static_models, observation, done)
        right_total_action, right_sample_actions_result = \
            self.get_side_total_action(self.right_models, self.right_static_models, observation, done)

        for env_index in range(self.env.num_envs):
            self.env.env_method("act", left_total_action[env_index], indices=env_index)
            self.env.env_method("inverted_act", right_total_action[env_index], indices=env_index)

        vecenv_data = self.env.env_method("commit")
        left_observation, right_observation, _, _ = list(zip(*vecenv_data))

        left_observation = np.array(left_observation)
        left_reward = self.env.env_method("calculate_left_reward")

        right_observation = np.array(right_observation)
        right_reward = self.env.env_method("calculate_right_reward")

        observation = [left_observation,] * len(self.left_models) + [right_observation,] * len(self.right_models)
        sample_actions_result = left_sample_actions_result + right_sample_actions_result
        reward = [left_reward,] * len(self.left_models) + [right_reward,] * len(self.right_models)

        # Используется, чтобы Monitor мог записать все необходимые логи. На самом деле данная функция не влияет на
        # состояние среды
        # TODO: Вообще, такой подход просто ужасен, но step не может сделать все сам, так как VecEnv не разрешает
        # вернуть несколько наград. Возможно, стоит это делать через info?
        _, _, done, info = self.env.step(left_total_action)

        return observation, reward, done, info, sample_actions_result


