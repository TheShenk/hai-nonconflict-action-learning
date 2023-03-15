from abc import abstractmethod

import gymnasium
import numpy as np

from hagl.base_types import HAGLType
from hagl.template import get_template, Template, DIMENSIONS_TEMPLATE_NAME

COORD_NAMES = ['x', 'y', 'z', 'w']
ARRAY_ACCESS_NAME = "array"

class Vector(HAGLType):

    def __add__(self, other):
        value = Vector()
        self_array_values = getattr(self, ARRAY_ACCESS_NAME)
        other_array_values = getattr(other, ARRAY_ACCESS_NAME)
        add_array_values = self_array_values+other_array_values

        for coord_n, coord_v in zip(COORD_NAMES, add_array_values):
            setattr(value, coord_n, coord_v)
        setattr(value, ARRAY_ACCESS_NAME, add_array_values)
        return value

class Velocity(Vector):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        value = Velocity()
        for dimension_index in range(t_dimensions_count):
            setattr(value, COORD_NAMES[dimension_index], gym_value[dimension_index])
        setattr(value, ARRAY_ACCESS_NAME, gym_value[:t_dimensions_count])
        return value

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return np.array(getattr(hagl_value, ARRAY_ACCESS_NAME))

class Position(Vector):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        value = Position()
        for dimension_index in range(t_dimensions_count):
            setattr(value, COORD_NAMES[dimension_index], gym_value[dimension_index])
        setattr(value, ARRAY_ACCESS_NAME, gym_value[:t_dimensions_count])
        return value

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return np.array(getattr(hagl_value, ARRAY_ACCESS_NAME))