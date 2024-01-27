import enum

from .base_types import HAGLType
from .exceptions import TemplateException

DIMENSIONS_TEMPLATE_NAME = "dimensions"

ENUM_GYM_CONVERSION_TEMPLATE_NAME = "enum_gym_conversion"
EnumGymConversion = enum.IntEnum("EnumGymConversion", ["Discrete", "Box"])

BOOL_GYM_CONVERSION_TEMPLATE_NAME = "bool_gym_conversion"
BoolGymConversion = enum.IntEnum("BoolGymConversion", ["Discrete", "Box"])

ANGLE_UNIT_TEMPLATE_NAME = "angle_unit"
AngleUnit = enum.IntEnum("AngleUnit", ["Radian", "Degree"])

DEFAULT_TEMPLATE_VALUES = {
    DIMENSIONS_TEMPLATE_NAME: 2,
    ENUM_GYM_CONVERSION_TEMPLATE_NAME: EnumGymConversion.Discrete,
    BOOL_GYM_CONVERSION_TEMPLATE_NAME: BoolGymConversion.Box,
    ANGLE_UNIT_TEMPLATE_NAME: AngleUnit.Radian
}


class Template(HAGLType):

    def __init__(self, template_name):
        self.template_name = template_name

    def name(self):
        return self.template_name

    def gym_type(self, template_values):
        return get_template(self, template_values)


def get_template(template_value, template_dict: dict):
    if isinstance(template_value, Template):
        template_name = template_value.name()
        if template_name in template_dict:
            return template_dict[template_name]
        else:
            raise TemplateException(f"Can't find template parameter with name {template_name}")
    return template_value


T = Template
