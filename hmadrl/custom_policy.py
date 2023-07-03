from abc import abstractmethod
from typing import Union, List, Optional, Dict, Tuple

import numpy as np
from ray.rllib import Policy, SampleBatch
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.typing import TensorStructType, TensorType, AgentID

import pygame


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
        assert len(obs_batch) == 1
        return [self.collect_action(obs) for obs in obs_batch], [], {"vf_preds": np.zeros((1,))}

    def get_weights(self):
        return {}

    def set_weights(self, weights) -> None:
        pass


class PyGamePolicy:

    def __init__(self, key_action_fn):
        self.key_action_fn = key_action_fn

    def collect_action(self, obs):
        if pygame.get_init():
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    return self.key_action_fn(event.key, obs)
        return self.key_action_fn(pygame.NOEVENT, obs)

    def __call__(self, obs):
        return self.collect_action(obs)


def ImitationPolicy(imitation_policy, model_class):
    return lambda observation_space, action_space, config: \
        _ImitationPolicy(imitation_policy, model_class, observation_space, action_space, config)


class _ImitationPolicy(CustomPolicy):

    def postprocess_trajectory(self, sample_batch: SampleBatch,
                               other_agent_batches: Optional[Dict[AgentID, Tuple["Policy", SampleBatch]]] = None,
                               episode: Optional["MultiAgentEpisode"] = None) -> SampleBatch:

        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS])
        # sample_batch[SampleBatch.ACTION_LOGP] = 1.0
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

    def _make_model(self, observation_space, action_space):
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            action_space, self.config["model"], framework="torch")
        self.model = ModelCatalog.get_model_v2(
            obs_space=observation_space,
            action_space=action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            framework="torch")

    def __init__(self, imitation_policy, model_cls, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.imitation_policy = imitation_policy
        self._make_model(observation_space, action_space)

    def collect_action(self, obs):
        action, _ = self.imitation_policy.predict(obs)
        return action
