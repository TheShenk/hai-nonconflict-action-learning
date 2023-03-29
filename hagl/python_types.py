import gymnasium.spaces
import numpy as np

from hagl.base_types import HAGLType
from hagl.template import get_template, Template, BOOL_GYM_CONVERSION_TEMPLATE_NAME, BoolGymConversion


class Float(HAGLType):

    @staticmethod
    def gym_type(template_values):
        return gymnasium.spaces.Box(-1.0, 1.0, (1,))

    @staticmethod
    def construct(gym_value, template_values):
        return gym_value[0]

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return np.array([hagl_value])


class Bool(HAGLType):

    @staticmethod
    def gym_type(template_values):
        conversion_type = get_template(Template(BOOL_GYM_CONVERSION_TEMPLATE_NAME), template_values)

        if conversion_type == BoolGymConversion.Discrete:
            return gymnasium.spaces.Discrete(2)
        elif conversion_type == BoolGymConversion.Box:
            return gymnasium.spaces.Box(0.0, 1.0, (1,))

    @staticmethod
    def construct(gym_value, template_values):
        return bool(gym_value)

    @staticmethod
    def deconstruct(hagl_value, template_values):
        return int(hagl_value)