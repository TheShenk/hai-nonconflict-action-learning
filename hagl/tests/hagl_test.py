import enum

import gymnasium.spaces.utils
import numpy as np

import hagl


def test_enum():

    class TestEnum(enum.IntEnum):
        a = enum.auto()
        b = enum.auto()
        c = enum.auto()

    class Space:
        field = TestEnum

    gymnasium_dict_space = hagl.compile_space(Space, {})
    gymnasium_space = gymnasium.spaces.utils.flatten_space(gymnasium_dict_space)
    assert isinstance(gymnasium_space, gymnasium.spaces.Box)
    assert gymnasium_space.shape == (3,)

    value = Space()
    value.field = TestEnum.b
    dict_value = hagl.deconstruct(Space, value, {})
    gymnasium_value = gymnasium.spaces.flatten(gymnasium_dict_space, dict_value)
    assert np.array_equal(gymnasium_value, np.array([0, 1, 0]))

    dict_value = gymnasium.spaces.unflatten(gymnasium_dict_space, gymnasium_value)
    value = hagl.construct(Space, dict_value, {})
    assert value.field == TestEnum.b
