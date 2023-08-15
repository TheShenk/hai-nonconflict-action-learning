from gymnasium.utils.env_checker import check_env
from pettingzoo.test import parallel_api_test
import hagl


class Observation:
    value = float


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


class TestMultiAgentHAGLEnv:

    def __init__(self):
        self.state = Observation()
        self.agents = ["add", "mul"]

    def reset(self):
        self.state.value = 0
        return {agent_id: self.state for agent_id in self.agents}, {}

    def step(self, action):
        self.state.value *= action["mul"].value
        self.state.value += action["add"].value
        return ({agent_id: self.state for agent_id in self.agents},
                {agent_id: 0.0 for agent_id in self.agents},
                {agent_id: False for agent_id in self.agents},
                {agent_id: {} for agent_id in self.agents}, {})

    def action_space(self, agent):
        return SingleAgentAction

    def observation_space(self, agent):
        return Observation


def test_with_parallel():

    env = TestMultiAgentHAGLEnv()
    env = hagl.HAGLParallelWrapper(env)

    obs = env.reset()
    assert obs["add"] == 0
    assert obs["mul"] == 0

    obs, _, _, _ = env.step({"add": [2], "mul": [3]})
    assert obs["add"] == 2
    assert obs["mul"] == 2

    obs, _, _, _ = env.step({"add": [0], "mul": [3]})
    assert obs["add"] == 6
    assert obs["mul"] == 6

    obs, _, _, _ = env.step({"add": [0], "mul": [0]})
    assert obs["add"] == 0
    assert obs["mul"] == 0

    parallel_api_test(env)
