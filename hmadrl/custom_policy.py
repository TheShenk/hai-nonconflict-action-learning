from abc import abstractmethod
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils import try_import_torch
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

torch, nn = try_import_torch()


class CustomPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.advantages = 0

    @abstractmethod
    def collect_action(self, obs):
        pass

    def compute_actions(self, obs_batch: Union[List[TensorStructType], TensorStructType],
                        state_batches: Optional[List[TensorType]] = None,
                        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
                        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
                        info_batch: Optional[Dict[str, list]] = None,
                        episodes: Optional[List["MultiAgentEpisode"]] = None,
                        explore: Optional[bool] = None,
                        timestep: Optional[int] = None, **kwargs) -> \
            Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        return [self.collect_action(obs) for obs in obs_batch], [], {"vf_preds": np.zeros((1,))}

    def get_weights(self):
        return {}

    def set_weights(self, weights) -> None:
        pass


def ImitationPolicy(imitation_policy, model_class, n_agents):
    return lambda observation_space, action_space, config: \
        _ImitationPolicy(imitation_policy, model_class, n_agents, observation_space, action_space, config)


class _ImitationPolicy(CustomPolicy):

    def __init__(self, imitation_policy, model_cls, n_agents, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.imitation_policy = imitation_policy
        self.n_agents = n_agents
        self._make_model(observation_space, action_space)

    def postprocess_trajectory(self, sample_batch: SampleBatch,
                               other_agent_batches: Optional[Dict[AgentID, Tuple["Policy", SampleBatch]]] = None,
                               episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:

        sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

        if other_agent_batches:
            agent_id = list(other_agent_batches.keys())[0]
            policy, batch = other_agent_batches[agent_id]

            for key in batch.keys():
                if key not in sample_batch:
                    sample_batch[key] = np.zeros_like(batch[key])

        return sample_batch

    def _make_model(self, observation_space, action_space):
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"], framework="torch")
        self.model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            framework="torch")

    def collect_action(self, obs):
        action, _ = self.imitation_policy.predict(obs)
        return action
