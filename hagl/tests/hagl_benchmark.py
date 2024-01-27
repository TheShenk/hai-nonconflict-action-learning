import hagl
import pettingzoo
from pettingzoo.utils import agent_selector
from pettingzoo.test import performance_benchmark


class BenchmarkAECHAGLEnv:

    def __init__(self, hagl_observation_space, hagl_action_space):
        self.hagl_observation_space = hagl_observation_space
        self.hagl_action_space = hagl_action_space

        gymnasium_observation_space = hagl.compile_space(hagl_observation_space, {})
        gymnasium_observation = gymnasium_observation_space.sample()
        self.observation = hagl.construct(hagl_observation_space, gymnasium_observation, {})

        self._agent_selector = None
        self.agent_selection = None

        self.agents = ["agent_0", "agent_1"]

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def reset(self, seed, options):
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        return self.observation

    def action_space(self, agent):
        return self.hagl_action_space

    def observation_space(self, agent):
        return self.hagl_observation_space


class BenchmarkAECEnv(pettingzoo.AECEnv):

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.gymnasium_observation_space = observation_space
        self.gymnasium_action_space = action_space

        self.observation = self.gymnasium_observation_space.sample()

        self._agent_selector = None
        self.agent_selection = None

        self.agents = ["agent_0", "agent_1"]

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def reset(self, seed=None, options=None):
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def step(self, action):
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        return self.observation

    def action_space(self, agent):
        return self.gymnasium_action_space

    def observation_space(self, agent):
        return self.gymnasium_observation_space


class ComplexSpace:
    value = float
    value_array = [float, 32]
    position = hagl.Position
    velocity = hagl.Limit(hagl.Velocity, low=-3.14, high=3.14)


env = BenchmarkAECHAGLEnv(ComplexSpace, ComplexSpace)
env = hagl.HAGLAECWrapper(env)
performance_benchmark(env)

env = BenchmarkAECEnv(env.observation_space("agent_0"), env.action_space("agent_0"))
performance_benchmark(env)
