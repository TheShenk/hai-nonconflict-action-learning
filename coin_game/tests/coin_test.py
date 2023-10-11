from coin_game import CoinGame
from pettingzoo.test import parallel_api_test


def test_parallel_api():
    env = CoinGame(2, 2)
    parallel_api_test(env)


def test_step_return():
    env = CoinGame(2, 2)
    env.reset()
    observation, reward, terminated, truncated, info = env.step({"agent_0": [0], "agent_1": [0]})
    assert len(observation["agent_0"]) == 2
    assert len(observation["agent_1"]) == 2

    assert observation["agent_0"][0][-1] == 0
    assert observation["agent_0"][1][-1] == 0
    assert observation["agent_1"][0][-1] == 0
    assert observation["agent_1"][1][-1] == 0
    assert reward["agent_0"] == 2
    assert reward["agent_1"] == 2

    observation, reward, terminated, truncated, info = env.step({"agent_0": [1], "agent_1": [0]})
    assert len(observation["agent_0"]) == 2
    assert len(observation["agent_1"]) == 2

    assert observation["agent_0"][0][-2] == 0
    assert observation["agent_0"][1][-2] == 0
    assert observation["agent_1"][0][-2] == 0
    assert observation["agent_1"][1][-2] == 0

    assert observation["agent_0"][0][-1] == 1
    assert observation["agent_0"][1][-1] == 0
    assert observation["agent_1"][0][-1] == 1
    assert observation["agent_1"][1][-1] == 0

    assert reward["agent_0"] == 3
    assert reward["agent_1"] == -1

    observation, reward, terminated, truncated, info = env.step({"agent_0": [0], "agent_1": [1]})
    assert len(observation["agent_0"]) == 2
    assert len(observation["agent_1"]) == 2

    assert observation["agent_0"][0][-2] == 1
    assert observation["agent_0"][1][-2] == 0
    assert observation["agent_1"][0][-2] == 1
    assert observation["agent_1"][1][-2] == 0

    assert observation["agent_0"][0][-1] == 0
    assert observation["agent_0"][1][-1] == 1
    assert observation["agent_1"][0][-1] == 0
    assert observation["agent_1"][1][-1] == 1

    assert reward["agent_0"] == -1
    assert reward["agent_1"] == 3

    observation, reward, terminated, truncated, info = env.step({"agent_0": [1], "agent_1": [1]})

    assert observation["agent_0"][0][-2] == 0
    assert observation["agent_0"][1][-2] == 1
    assert observation["agent_1"][0][-2] == 0
    assert observation["agent_1"][1][-2] == 1

    assert observation["agent_0"][0][-1] == 1
    assert observation["agent_0"][1][-1] == 1
    assert observation["agent_1"][0][-1] == 1
    assert observation["agent_1"][1][-1] == 1

    assert reward["agent_0"] == 0
    assert reward["agent_1"] == 0
