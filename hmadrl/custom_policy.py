from abc import abstractmethod
from typing import Union, List, Optional, Dict, Tuple

from ray.rllib import Policy
from ray.rllib.utils.typing import TensorStructType, TensorType

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
        return [self.collect_action(obs) for obs in obs_batch], [], {}

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


def ImitationPolicy(imitation_policy):
    return lambda observation_space, action_space, config: \
        _ImitationPolicy(imitation_policy, observation_space, action_space, config)


class _ImitationPolicy(CustomPolicy):

    def __init__(self, imitation_policy, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.imitation_policy = imitation_policy

    def collect_action(self, obs):
        action, _ = self.imitation_policy.predict(obs)
        print(action)
        return action
