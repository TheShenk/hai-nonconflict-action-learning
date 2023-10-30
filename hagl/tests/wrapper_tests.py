from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test, api_test
from pettingzoo.utils import agent_selector

import hagl

ValueT = hagl.Limit(float, low=float("-inf"), high=float("inf"))


class Observation:
    value = ValueT


class Action:
    add = float
    mul = float


class TestHAGLEnv:

    def __init__(self):
        self.state = Observation()
        self.observation_space = Observation
        self.action_space = Action

    def reset(self):
        self.state.value = 0
        return self.state, {}, {}

    def step(self, action):
        self.state.value *= action.mul
        self.state.value += action.add
        return self.state, 0.0, True, False, {}, {}


def test_with_gym():
    env = TestHAGLEnv()
    env = hagl.HAGLWrapper(env)

    value, _ = env.reset()
    assert value == 0

    value, _, _, _, _ = env.step([2, 3])
    assert value == 2

    value, _, _, _, _ = env.step([0, 3])
    assert value == 6

    value, _, _, _, _ = env.step([0, 0])
    assert value == 0

    check_env(env)


class SingleAgentAction:
    value = float


class TestParallelHAGLEnv:

    def __init__(self):
        self.state = Observation()
        self.agents = ["add", "mul"]

    def reset(self, seed, options):
        self.state.value = 0
        return {agent_id: self.state for agent_id in self.agents}, {agent_id: {} for agent_id in self.agents}, {}

    def step(self, action):
        self.state.value *= action["mul"].value
        self.state.value += action["add"].value
        return ({agent_id: self.state for agent_id in self.agents},
                {agent_id: 0.0 for agent_id in self.agents},
                {agent_id: False for agent_id in self.agents},
                {agent_id: False for agent_id in self.agents},
                {agent_id: {} for agent_id in self.agents}, {})

    def action_space(self, agent):
        return SingleAgentAction

    def observation_space(self, agent):
        return Observation


def test_with_parallel():
    env = TestParallelHAGLEnv()
    env = hagl.HAGLParallelWrapper(env)

    obs, info = env.reset()
    assert obs["add"] == 0
    assert obs["mul"] == 0
    assert info == {"add": {}, "mul": {}}

    obs, _, _, _, _ = env.step({"add": [2], "mul": [3]})
    assert obs["add"] == 2
    assert obs["mul"] == 2

    obs, _, _, _, _ = env.step({"add": [0], "mul": [3]})
    assert obs["add"] == 6
    assert obs["mul"] == 6

    obs, _, _, _, _ = env.step({"add": [0], "mul": [0]})
    assert obs["add"] == 0
    assert obs["mul"] == 0

    parallel_api_test(env)


class TestAECHAGLEnv:

    def __init__(self):
        self._agent_selector = None
        self.agent_selection = None

        self.state = Observation()
        self.agents = ["add", "mul"]

        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    def reset(self, seed, options):
        self.state.value = 0
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        # self.observations = {agent: NONE for agent in self.agents}

    def step(self, action):
        if self.agent_selection == "add":
            self.state.value += action.value
        elif self.agent_selection == "mul":
            self.state.value *= action.value
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent):
        return self.state

    def action_space(self, agent):
        return SingleAgentAction

    def observation_space(self, agent):
        return Observation


def test_with_aec():
    env = TestAECHAGLEnv()
    env = hagl.HAGLAECWrapper(env)

    env.reset()
    obs = env.observe("add")
    assert obs == 0

    env.step([2])
    obs = env.observe("add")
    assert obs == 2

    env.step([3])
    obs = env.observe("mul")
    assert obs == 6

    env.step([-6])
    obs = env.observe("add")
    assert obs == 0

    api_test(env)
