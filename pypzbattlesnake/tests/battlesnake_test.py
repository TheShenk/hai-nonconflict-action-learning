import numpy as np
from pettingzoo.test import parallel_api_test

from pypzbattlesnake.common import Action
from pypzbattlesnake.env import BattleSnake

REWARD_STEP = 1 / 18
FLOAT_EPS = 0.00001


def count_cell_colors(obs):

    snake_colors = {}
    for snake, snake_obs in obs.items():
        colors = [0, ] * 5
        for cell in snake_obs[:81]:
            if cell:
                colors[int(cell) - 1] += 1
        snake_colors[snake] = tuple(colors)
    assert len(set(snake_colors.values())) == 1
    return next(iter(snake_colors.values()))


def test_parallel_api():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    parallel_api_test(env)


def test_reset_position():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)

    obs, _ = env.reset()
    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert env.snakes["snake_0_1"].head() == (7, 1)
    assert env.snakes["snake_1_0"].head() == (7, 7)
    assert env.snakes["snake_1_1"].head() == (1, 7)

    field = np.reshape(obs["snake_0_0"][:81], (9, 9))
    assert field[1][1] == 2
    assert field[7][1] == 3
    assert field[7][7] == 4
    assert field[1][7] == 5

    directions = obs["snake_0_0"][81:]
    assert directions[0] == Action.NONE
    assert directions[1] == Action.NONE
    assert directions[2] == Action.NONE
    assert directions[3] == Action.NONE

    obs, _ = env.reset()
    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert env.snakes["snake_0_1"].head() == (7, 1)
    assert env.snakes["snake_1_0"].head() == (7, 7)
    assert env.snakes["snake_1_1"].head() == (1, 7)

    field = np.reshape(obs["snake_0_0"][:81], (9, 9))
    assert field[1][1] == 2
    assert field[7][1] == 3
    assert field[7][7] == 4
    assert field[1][7] == 5

    directions = obs["snake_0_0"][81:]
    assert directions[0] == Action.NONE
    assert directions[1] == Action.NONE
    assert directions[2] == Action.NONE
    assert directions[3] == Action.NONE

    assert env.snakes["snake_0_0"].color == 2
    assert env.snakes["snake_0_1"].color == 3
    assert env.snakes["snake_1_0"].color == 4
    assert env.snakes["snake_1_1"].color == 5


def test_step():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert field[1][1] == 2
    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step({"snake_0_0": [1, 0, 0, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
                                                              "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))

    assert env.snakes["snake_0_0"].head() == (1, 0)
    assert field[1][0] == 2
    assert abs(reward["snake_0_0"] - -REWARD_STEP) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 1, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (0, 0)
    assert field[0][0] == 2
    assert directions[0] == Action.LEFT
    assert abs(reward["snake_0_0"] - -REWARD_STEP) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 1, 0, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (0, 1)
    assert field[0][1] == 2
    assert directions[0] == Action.DOWN
    assert abs(reward["snake_0_0"] - REWARD_STEP) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 1, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert field[1][1] == 2
    assert directions[0] == Action.RIGHT
    assert abs(reward["snake_0_0"] - REWARD_STEP) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]


def test_border_collision():

    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert field[1][1] == 2
    assert directions[0] == Action.NONE

    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step({"snake_0_0": [1, 0, 0, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
                                                              "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (1, 0)
    assert field[1][0] == 2
    assert directions[0] == Action.UP
    assert abs(reward["snake_0_0"] - -REWARD_STEP) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]
    assert count_cell_colors(observation) == (3, 1, 1, 1, 1)

    observation, reward, terminated, truncated, _ = env.step({"snake_0_0": [1, 0, 0, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
                                                              "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})

    assert env.snakes.get("snake_0_0", None) is None
    assert count_cell_colors(observation) == (3, 0, 1, 1, 1)

    assert abs(reward["snake_0_0"] - -1.0) < FLOAT_EPS
    assert terminated["snake_0_0"]
    assert not truncated["snake_0_0"]


def test_food_collision():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (1, 1)
    assert field[1][1] == 2
    assert directions[0] == Action.NONE
    env.food = {(2, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 1, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_0_0"][:81], (9, 9))
    directions = observation["snake_0_0"][81:]

    assert env.snakes["snake_0_0"].head() == (2, 1)
    assert env.snakes["snake_0_0"].tail() == (1, 1)
    assert not field[2][1] == 1
    assert field[2][1] == 2
    assert field[1][1] == 2
    assert directions[0] == Action.RIGHT
    assert abs(reward["snake_0_0"] - 1.0) < FLOAT_EPS
    assert not terminated["snake_0_0"]
    assert not truncated["snake_0_0"]
    assert count_cell_colors(observation) == (3, 2, 1, 1, 1)


def test_another_collision():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()

    env.snakes["snake_1_0"].body = [(2, 0), (2, 1), (2, 2)]
    env.snakes["snake_1_0"].action = Action.DOWN
    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 1, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_1_0"][:81], (9, 9))
    directions = observation["snake_1_0"][81:]

    assert env.snakes.get("snake_0_0", None) is None
    assert count_cell_colors(observation) == (3, 0, 1, 3, 1)

    assert field[2][1] == 4
    assert field[2][2] == 4
    assert field[2][3] == 4
    assert directions[0] == Action.NONE
    assert abs(reward["snake_0_0"] - -1.0) < FLOAT_EPS
    assert abs(reward["snake_1_0"] - 1.0) < FLOAT_EPS
    assert terminated["snake_0_0"]
    assert not truncated["snake_0_0"]


def test_teammate_collision():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()

    env.snakes["snake_0_1"].body = [(2, 0), (2, 1), (2, 2)]
    env.snakes["snake_0_1"].action = Action.DOWN
    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 1, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})
    field = np.reshape(observation["snake_1_0"][:81], (9, 9))
    directions = observation["snake_1_0"][81:]

    assert env.snakes.get("snake_0_0", None) is None
    assert count_cell_colors(observation) == (3, 0, 3, 1, 1)

    assert field[2][1] == 3
    assert field[2][2] == 3
    assert field[2][3] == 3
    assert directions[0] == Action.NONE
    assert abs(reward["snake_0_0"] - -1.0) < FLOAT_EPS
    assert abs(reward["snake_0_1"] - -1.0) < FLOAT_EPS
    assert terminated["snake_0_0"]
    assert not truncated["snake_0_0"]


def test_head_collision():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()

    env.snakes["snake_1_0"].body = [(4, 1), (3, 1)]
    env.snakes["snake_1_0"].action = Action.LEFT
    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 1, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 1, 0, 0], "snake_1_1": [0, 0, 0, 0, 1]})

    assert env.snakes.get("snake_0_0", None) is None
    assert env.snakes.get("snake_1_0", None) is None
    assert count_cell_colors(observation) == (3, 0, 1, 0, 1)

    assert abs(reward["snake_0_0"] - -1.0) < FLOAT_EPS
    assert abs(reward["snake_1_0"] - -1.0) < FLOAT_EPS
    assert terminated["snake_0_0"]
    assert terminated["snake_1_0"]
    assert not truncated["snake_0_0"]
    assert not truncated["snake_1_0"]


def test_self_collision():
    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()

    env.snakes["snake_0_0"].body = [(0, 1), (1, 1), (2, 1), (2, 2), (1, 2)]
    env.snakes["snake_0_0"].action = Action.LEFT
    env.food = {(6, 1), (1, 6), (6, 6)}

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [1, 0, 0, 0, 0], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})

    assert env.snakes.get("snake_0_0", None) is None
    assert count_cell_colors(observation) == (3, 0, 1, 1, 1)

    assert abs(reward["snake_0_0"] - -1.0) < FLOAT_EPS
    assert terminated["snake_0_0"]
    assert not truncated["snake_0_0"]


def test_zero_health():

    env = BattleSnake(2, 2, (9, 9), 3, 5, 1.0)
    observation, _, = env.reset()
    env.food = {(6, 1), (1, 6), (6, 6)}

    assert count_cell_colors(observation) == (3, 1, 1, 1, 1)

    for i in range(99):
        observation, reward, terminated, truncated, _ = env.step(
            {"snake_0_0": [0, 0, 0, 0, 1], "snake_0_1": [0, 0, 0, 0, 1],
             "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})

    assert count_cell_colors(observation) == (22, 1, 1, 1, 1)

    observation, reward, terminated, truncated, _ = env.step(
        {"snake_0_0": [0, 0, 0, 0, 1], "snake_0_1": [0, 0, 0, 0, 1],
         "snake_1_0": [0, 0, 0, 0, 1], "snake_1_1": [0, 0, 0, 0, 1]})

    assert count_cell_colors(observation) == (23, 0, 0, 0, 0)
