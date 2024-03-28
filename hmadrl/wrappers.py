from __future__ import annotations

import functools
import warnings

import gymnasium.spaces
import pettingzoo
from pettingzoo.utils.env import AgentID, ObsType, ActionType


class FlattenAction(pettingzoo.utils.BaseParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

        try:
            self.render_mode = env.render_mode
        except AttributeError:
            warnings.warn(f"The base environment `{env}` does not have a `render_mode` defined.")

    @functools.lru_cache
    def action_space(self, agent):
        return gymnasium.spaces.flatten_space(super().action_space(agent))

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        actions = {agent: gymnasium.spaces.unflatten(self.env.action_space(agent), act)
                   for agent, act in actions.items()}
        return super().step(actions)


class FlattenObservation(pettingzoo.utils.BaseParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

        try:
            self.render_mode = env.render_mode
        except AttributeError:
            warnings.warn(f"The base environment `{env}` does not have a `render_mode` defined.")

    @functools.lru_cache
    def observation_space(self, agent):
        return gymnasium.spaces.flatten_space(super().observation_space(agent))

    def reset(self, seed: int | None = None, options: dict | None = None) \
            -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        observation, info = super().reset(seed, options)
        observation = {agent: gymnasium.spaces.flatten(self.env.observation_space(agent), obs)
                       for agent, obs in observation.items()}
        return observation, info

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        observation, reward, terminated, truncated, info = super().step(actions)
        observation = {agent: gymnasium.spaces.flatten(self.env.observation_space(agent), obs)
                       for agent, obs in observation.items()}

        return observation, reward, terminated, truncated, info


def Flatten(env): return FlattenObservation(FlattenAction(env))
