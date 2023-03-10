from abc import abstractmethod

import gymnasium
import numpy as np

from hagl.base_types import HAGLType, compile_type
from hagl.template import get_template, Template, DIMENSIONS_TEMPLATE_NAME


class Velocity(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray):
        # TODO: Используя x, y и z поддерживаем только 2D и 3D, добавить также обращение через массив?
        value = Velocity()
        coord_names = ['x', 'y', 'z', 'w']
        for coord_n, coord_v in zip(coord_names, gym_value):
            setattr(value, coord_n, coord_v)
        return value

class Position(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @staticmethod
    def construct(gym_value: np.ndarray):
        # TODO: Используя x, y и z поддерживаем только 2D и 3D, добавить также обращение через массив?
        value = Position()
        coord_names = ['x', 'y', 'z', 'w']
        for coord_n, coord_v in zip(coord_names, gym_value):
            setattr(value, coord_n, coord_v)
        return value