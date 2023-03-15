import enum

import gymnasium
import numpy as np

import hagl
from hagl import HAGLType
from hagl.template import EnumGymConversion, ENUM_GYM_CONVERSION_TEMPLATE_NAME, get_template, Template


class Enum(HAGLType):

    def __init__(self, python_enum: enum.EnumMeta):
        self.python_enum = python_enum
        self.conversion_type = EnumGymConversion.Discrete

    def gym_type(self, template_values):
        conversion_type = get_template(Template(ENUM_GYM_CONVERSION_TEMPLATE_NAME), template_values)

        if conversion_type == EnumGymConversion.Discrete:
            return gymnasium.spaces.Discrete(len(self.python_enum))
        elif conversion_type == EnumGymConversion.Box:
            return gymnasium.spaces.Box(low=1.0, high=len(self.python_enum))

    def construct(self, gym_value, template_values):
        conversion_type = get_template(Template(ENUM_GYM_CONVERSION_TEMPLATE_NAME), template_values)

        value = gym_value + 1
        if conversion_type == EnumGymConversion.Box:
            value = int(gym_value)

        return self.python_enum(value)

    def deconstruct(self, hagl_value, template_values):
        conversion_type = get_template(Template(ENUM_GYM_CONVERSION_TEMPLATE_NAME), template_values)

        if conversion_type == EnumGymConversion.Discrete:
            return int(hagl_value - 1)
        elif conversion_type == EnumGymConversion.Box:
            return int(hagl_value)