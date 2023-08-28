import numpy as np
import pytest
from gymnasium.spaces import Discrete
from numpy.typing import NDArray
from pettingzoo.test import parallel_api_test
from gymnasium.spaces.utils import unflatten

from hagl import HAGLParallelWrapper
from pypzbattlesnake.env import BattleSnake


@unflatten.register(Discrete)
def _unflatten_discrete(space: Discrete, x: NDArray[np.int64]) -> np.int64:
    nonzero = np.nonzero(x)
    if len(nonzero[0]) == 0:
        return space.start
    return space.start + nonzero[0][0]


def test_parallel_api():

    env = BattleSnake(4)
    env = HAGLParallelWrapper(env)
    parallel_api_test(env)
