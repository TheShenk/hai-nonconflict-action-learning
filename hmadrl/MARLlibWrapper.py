from __future__ import annotations

import functools
from typing import Tuple

import gym
import gymnasium.spaces
import numpy as np
import pettingzoo
import supersuit
from pettingzoo.utils.env import AgentID, ActionType, ObsType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from hagl.convert_space import convert_space


class TimeLimit(pettingzoo.utils.BaseParallelWrapper):

    def __init__(self, env, max_episode_steps):
        super().__init__(env)
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = None

    def reset(self, seed: int | None = None, options: dict | None = None):
        res = super().reset(seed, options)
        self.elapsed_steps = 0
        return res

    def step(self, actions):
        observation, reward, terminated, truncated, info = super().step(actions)
        self.elapsed_steps += 1
        if self.elapsed_steps >= self.max_episode_steps:
            terminated = {agent: True for agent in terminated}
            # truncated = {agent: True for agent in truncated}
        return observation, reward, terminated, truncated, info


class Flatten(pettingzoo.utils.BaseParallelWrapper):

    def __init__(self, env):
        super().__init__(env)

    @functools.lru_cache
    def observation_space(self, agent):
        return gymnasium.spaces.flatten_space(super().observation_space(agent))

    @functools.lru_cache
    def action_space(self, agent):
        return gymnasium.spaces.flatten_space(super().action_space(agent))

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
        actions = {agent: gymnasium.spaces.unflatten(self.env.action_space(agent), act)
                   for agent, act in actions.items()}
        observation, reward, terminated, truncated, info = super().step(actions)
        observation = {agent: gymnasium.spaces.flatten(self.env.observation_space(agent), obs)
                       for agent, obs in observation.items()}

        return observation, reward, terminated, truncated, info


class Float(pettingzoo.utils.BaseParallelWrapper):

    def __init__(self, env, dtype=np.float64):
        super().__init__(env)
        self.dtype = dtype

    @functools.lru_cache
    def observation_space(self, agent):
        space: gymnasium.spaces.Box = super().observation_space(agent)
        return gymnasium.spaces.Box(low=space.low, high=space.high, dtype=self.dtype)

    @functools.lru_cache
    def action_space(self, agent):
        space: gymnasium.spaces.Box = super().action_space(agent)
        return gymnasium.spaces.Box(low=space.low, high=space.high, dtype=self.dtype)


class MARLlibWrapper(MultiAgentEnv):

    def __init__(self, env: pettingzoo.ParallelEnv, max_episode_len, policy_mapping_info):
        super().__init__()

        self.env = env
        self.possible_agents = self.env.possible_agents
        self.num_agents = len(self.possible_agents)
        self.observation_space = gym.spaces.Dict({"obs": convert_space(self.env.observation_space(self.possible_agents[0]))})
        self.action_space = convert_space(self.env.action_space(self.possible_agents[0]))

        self.max_episode_len = max_episode_len
        self.policy_mapping_info = policy_mapping_info

    def reset(self) -> MultiAgentDict:
        observation, info = self.env.reset()
        observation = {agent: {"obs": observation[agent]} for agent in observation}
        return observation

    def step(self, action: MultiAgentDict) -> (
            Tuple)[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = {agent: {"obs": observation[agent]} for agent in observation}
        done = {agent: terminated[agent] or truncated[agent] for agent in terminated}
        done["__all__"] = np.all(list(done.values()))
        return observation, reward, done, info

    def close(self):
        self.env.close()

    def render(self, mode=None):
        self.env.render()
        return True

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.num_agents,
            "episode_limit": self.max_episode_len,
            "policy_mapping_info": self.policy_mapping_info
        }
        return env_info

    @property
    def agents(self):
        return self.env.agents

class CoopMARLlibWrapper(MARLlibWrapper):

    def step(self, action: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = {agent: {"obs": observation[agent]} for agent in action.keys()}
        coop_reward = sum([reward[agent] for agent in observation])
        reward = {agent: coop_reward for agent in observation}
        done = {agent: terminated[agent] or truncated[agent] for agent in action.keys()}
        total_done = np.all(list(done.values()))
        done["__all__"] = total_done
        return observation, reward, done, info
