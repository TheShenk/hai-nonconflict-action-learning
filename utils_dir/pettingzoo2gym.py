import gymnasium as gym
import numpy as np
from pettingzoo import ParallelEnv

class AgentEnv(gym.Wrapper):

    def __init__(self, agent_id, env: ParallelEnv):
        super().__init__(env)
        self.agent_id = agent_id
        self.real_observation_space = env.observation_space(agent_id)
        self.real_action_space = env.action_space(agent_id)

        self.observation_space = gym.spaces.flatten_space(self.real_observation_space)
        self.action_space = gym.spaces.flatten_space(self.real_action_space)

    def reset(
        self, *, seed = None, options = None
    ):
        observation = self.env.reset(seed, options)
        return observation[self.agent_id]


class PettingZoo2Gym(gym.Env):

    def step(self, action):

        actions_by_agents = gym.spaces.unflatten(self.total_action_space, action)
        observation, reward, terminated, truncated, info = self.env.step(actions_by_agents)
        observation = gym.spaces.flatten(self.total_observation_space, observation)

        return observation, reward[self.env.possible_agents[0]], np.any(list(terminated.values())) or np.any(list(truncated.values())), info[self.env.possible_agents[0]]

    def reset(self):
        observation = self.env.reset()
        observation = gym.spaces.flatten(self.total_observation_space, observation)
        return observation

    def render(self, mode="human"):
        pass

    def __init__(self, env: ParallelEnv):
        self.env = env
        self.num_envs = len(env.possible_agents)
        self.total_action_space = gym.spaces.Dict(env.action_spaces)
        self.total_observation_space = gym.spaces.Dict(env.observation_spaces)

        self.action_space = gym.spaces.flatten_space(self.total_action_space)
        self.observation_space = gym.spaces.flatten_space(self.total_observation_space)

    def agent_envs(self):
        return [AgentEnv(agent_id, self.env) for agent_id in self.env.possible_agents]


