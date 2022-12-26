from typing import List

import numpy as np
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from multiagent.action_combiners import BOX_COMBINER
from multiagent.multi_agent_proxy import MultiAgentProxy
from multiagent.multi_model_agent import MultiModelAgent


class TwoSideModelAgent(MultiModelAgent):

    def __init__(self,
                 env: VecEnv,
                 left_models: List[MultiAgentProxy],
                 right_models: List[MultiAgentProxy],
                 actions_combiner=BOX_COMBINER):

        models = left_models + right_models
        super().__init__(env, models, actions_combiner=actions_combiner)

        self.left_models = left_models
        self.right_models = right_models

    def collect_step_info(self, observation, done):

        left_sample_actions_result = [model.sample_action() for model in self.left_models]
        right_sample_actions_result = [model.sample_action() for model in self.right_models]

        left_actions = tuple(map(lambda sar: sar[0], left_sample_actions_result))
        right_actions = tuple(map(lambda sar: sar[0], right_sample_actions_result))

        left_total_action = self.actions_combiner(left_actions)
        right_total_action = self.actions_combiner(right_actions)

        for env_index in range(self.env.num_envs):
            self.env.env_method("act", left_total_action[env_index], indices=env_index)
            self.env.env_method("inverted_act", right_total_action[env_index], indices=env_index)

        vecenv_data = self.env.env_method("commit")

        left_observation, right_observation, done, info = list(zip(*vecenv_data))
        left_observation = np.array(left_observation)
        right_observation = np.array(right_observation)

        left_reward = self.env.env_method("calculate_left_reward")
        right_reward = self.env.env_method("calculate_right_reward")

        observation = [left_observation,] * len(self.left_models) + [right_observation,] * len(self.right_models)
        sample_actions_result = left_sample_actions_result + right_sample_actions_result
        reward = [left_reward,] * len(self.left_models) + [right_reward,] * len(self.right_models)
        done = np.array(done)

        return observation, reward, done, info, sample_actions_result


