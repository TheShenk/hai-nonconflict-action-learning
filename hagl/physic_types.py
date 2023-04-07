import math

import gymnasium
import numpy as np

from hagl.base_types import HAGLType
from hagl.exceptions import PhysicException
from hagl.template import get_template, Template, DIMENSIONS_TEMPLATE_NAME, ANGLE_UNIT_TEMPLATE_NAME, AngleUnit
from hagl.python_types import Float

COORD_NAMES = ['x', 'y', 'z', 'w']

class Vector(HAGLType):

    @staticmethod
    def gym_type(template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(t_dimensions_count,))

    @classmethod
    def construct(cls, gym_value: np.ndarray, template_values):
        t_dimensions_count = get_template(Template(DIMENSIONS_TEMPLATE_NAME), template_values)

        value = cls()
        value.array = np.array(gym_value[:t_dimensions_count])

        return value

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return np.array(hagl_value.array)

    def __add__(self, other):
        value = Vector()

        if isinstance(other, Vector):
            add_array_values = self.array+other.array
        elif isinstance(other, np.ndarray):
            add_array_values = self.array + other
        elif isinstance(other, list):
            add_array_values = self.array + np.array(other)
        else:
            raise PhysicException(f"Unknown object to add to vector: {other}")

        value.array = add_array_values
        return value

    def __mul__(self, other):
        value = Vector()

        if isinstance(other, Vector):
            add_array_values = self.array * other.array
        elif isinstance(other, np.ndarray):
            add_array_values = self.array * other
        elif isinstance(other, list):
            add_array_values = self.array * np.array(other)
        else:
            raise PhysicException(f"Unknown object to mul with vector: {other}")

        value.array = add_array_values
        return value

    def __setattr__(self, key, value):
        if key in COORD_NAMES:
            key_index = COORD_NAMES.index(key)
            self.array[key_index] = value
        else:
            super().__setattr__(key, value)

    def __getattr__(self, item):
        if item in COORD_NAMES:
            item_index = COORD_NAMES.index(item)
            return self.array[item_index]
        else:
            return super().__getattribute__(item)

    def norm(self):
        return np.linalg.norm(self.array)

class Velocity(Vector):
    pass


class Position(Vector):
    pass

class Angle(HAGLType):
    @staticmethod
    def gym_type(template_values):
        angle_unit = get_template(Template(ANGLE_UNIT_TEMPLATE_NAME), template_values)
        if angle_unit == AngleUnit.Radian:
            return gymnasium.spaces.Box(low=-math.pi, high=math.pi, shape=(1,))
        elif angle_unit == AngleUnit.Degree:
            return gymnasium.spaces.Box(low=--180.0, high=180.0, shape=(1,))

    @staticmethod
    def construct(gym_value: np.ndarray, template_values):
        return gym_value[0]

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return np.array([hagl_value])

class AngleVelocity(Float):
    pass