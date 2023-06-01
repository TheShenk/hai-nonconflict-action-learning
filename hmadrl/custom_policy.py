from abc import abstractmethod
from typing import Union, List, Optional, Dict, Tuple

from ray.rllib import Policy
from ray.rllib.utils.typing import TensorStructType, TensorType

import pygame


class HumanPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.advantages = 0

    @abstractmethod
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
        return [self.collect_action() for _ in obs_batch], [], {}

    def get_weights(self):
        return {}

    def set_weights(self, weights) -> None:
        pass

class _PyGamePolicy(HumanPolicy):

    def __init__(self, key_action_fn, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)
        self.key_action_fn = key_action_fn

    def compute_single_action(self, obs):
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                return self.key_action_fn(event.key, obs)
        return self.key_action_fn(pygame.NOEVENT, obs)

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
        return [self.compute_single_action(obs) for obs in obs_batch], [], {}


def PyGamePolicy(key_action_dict):
    return lambda observation_space, action_space, config: \
        _PyGamePolicy(key_action_dict, observation_space, action_space, config)
