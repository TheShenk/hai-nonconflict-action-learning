from abc import abstractmethod

import gymnasium
import numpy as np

from hagl.base_types import HAGLType
from hagl.template import get_template, Template, DIMENSIONS_TEMPLATE_NAME

COORD_NAMES = ['x', 'y', 'z', 'w']

class Velocity(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        # TODO: Используя x, y, z, ... поддерживаем ограниченное число измерений, добавить также обращение через массив?
        value = Velocity()
        for dimension_index in range(t_dimensions_count):
            setattr(value, COORD_NAMES[dimension_index], gym_value[dimension_index])
        return value

    @staticmethod
    def deconstruct(hagl_value, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        value = []
        for coord_n in COORD_NAMES[:t_dimensions_count]:
            coord_v = getattr(hagl_value, coord_n)
            value.append(coord_v)

        return np.array(value)

class Position(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        #TODO: Используя x, y, z, ... поддерживаем ограниченное число измерений, добавить также обращение через массив?
        value = Velocity()
        for dimension_index in range(t_dimensions_count):
            setattr(value, COORD_NAMES[dimension_index], gym_value[dimension_index])
        return value

    @staticmethod
    def deconstruct(hagl_value, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        value = []
        for coord_n in COORD_NAMES[:t_dimensions_count]:
            coord_v = getattr(hagl_value, coord_n)
            value.append(coord_v)

        return np.array(value)