from abc import abstractmethod

import gymnasium
import numpy as np

from hagl.base_types import HAGLType, compile_type
from hagl.template import get_template, Template, DIMENSIONS_TEMPLATE_NAME

COORD_NAMES = ['x', 'y', 'z', 'w']

class Velocity(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray):
        # TODO: Используя x, y, z, ... поддерживаем ограниченное число измерений, добавить также обращение через массив?
        value = Velocity()
        for coord_n, coord_v in zip(COORD_NAMES, gym_value):
            setattr(value, coord_n, coord_v)
        return value

    @staticmethod
    def deconstruct(hagl_value):
        value = []
        for coord_n in COORD_NAMES:
            if hasattr(hagl_value, coord_n):
                coord_v = getattr(hagl_value, coord_n)
                value.append(coord_v)

        return np.array(value)

class Position(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray):
        # TODO: Используя x, y, z, ... поддерживаем ограниченное число измерений, добавить также обращение через массив?
        value = Position()
        for coord_n, coord_v in zip(COORD_NAMES, gym_value):
            setattr(value, coord_n, coord_v)
        return value

    @staticmethod
    def deconstruct(hagl_value):
        value = []
        for coord_n in COORD_NAMES:
            if hasattr(hagl_value, coord_n):
                coord_v = getattr(hagl_value, coord_n)
                value.append(coord_v)

        return np.array(value)