from typing import Tuple

import gym
import numpy as np
import pettingzoo
import supersuit
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict

from hagl.convert_space import convert_space


class MARLlibWrapper(MultiAgentEnv):

    def __init__(self, env: pettingzoo.ParallelEnv, max_episode_len, policy_mapping_info):

        super().__init__()

        # keep obs and action dim same across agents
        # pad_action_space_v0 will auto mask the padding actions
        env = supersuit.pad_observations_v0(env)
        env = supersuit.pad_action_space_v0(env)
        env = supersuit.black_death_v3(env)

        self.env = env
        self.agents = env.possible_agents
        self.num_agents = len(self.agents)
        self.observation_space = gym.spaces.Dict({"obs": convert_space(self.env.observation_space(self.agents[0]))})
        self.action_space = convert_space(self.env.action_space(self.agents[0]))

        self.max_episode_len = max_episode_len
        self.policy_mapping_info = policy_mapping_info

    def reset(self) -> MultiAgentDict:
        observation, info = self.env.reset()
        observation = {agent: {"obs": observation[agent]} for agent in self.agents}
        return observation

    def step(self, action: MultiAgentDict) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = {agent: {"obs": observation[agent]} for agent in action.keys()}
        done = {agent: terminated[agent] or truncated[agent] for agent in action.keys()}
        total_done = np.all(list(done.values()))
        done["__all__"] = total_done
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


class CoopMARLlibWrapper(MARLlibWrapper):

    def step(self, action: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        observation, reward, done, info = self.env.step(action)
        observation = {agent: {"obs": observation[agent]} for agent in action.keys()}
        coop_reward = sum([reward[agent] for agent in observation])
        reward = {agent: coop_reward for agent in observation}
        return observation, reward, done, info
