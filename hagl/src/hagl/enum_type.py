import enum

import gymnasium

from .base_types import HAGLType
from .template import EnumGymConversion, ENUM_GYM_CONVERSION_TEMPLATE_NAME, get_template, Template
from .exceptions import EnumException


class Enum(HAGLType):

    def __init__(self, python_enum: enum.EnumMeta):
        self.python_enum = python_enum
        self.conversion_type = None

    def gym_type(self, template_values):
        conversion_type = get_template(Template(ENUM_GYM_CONVERSION_TEMPLATE_NAME), template_values)

        if conversion_type == EnumGymConversion.Discrete:
            return gymnasium.spaces.Discrete(len(self.python_enum))
        elif conversion_type == EnumGymConversion.Box:
            return gymnasium.spaces.Box(low=0.0, high=len(self.python_enum)-1)
        else:
            raise EnumException(f"Unknown enum-gym conversion type: {conversion_type}")

    def construct(self, gym_value, template_values):
        value = int(gym_value + 1)
        return self.python_enum(value)

    def deconstruct(self, hagl_value, template_values):
        assert hagl_value in iter(self.python_enum), f"Received a value {hagl_value} that does not belong to the enum"
        return int(hagl_value - 1)
