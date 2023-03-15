from collections import OrderedDict
import enum

import gymnasium

import hagl
from hagl import HAGLType


def is_base_hagl_type(val):
    if type(val) == type:
        return issubclass(val, HAGLType)
    else:
        return issubclass(type(val), HAGLType)


def try_as_syntax_shugar(val):
    if isinstance(val, list) and len(val) == 2:
        return hagl.composition_types.Array(val[0], val[1])
    elif isinstance(val, enum.EnumMeta):
        return hagl.enum_type.Enum(val)
    return val


def isfunction(obj):
    return callable(obj) and not isinstance(obj, type)

def allowed_vars(hagl_type):
    type_vars = vars(hagl_type)
    return {name: value for name, value in type_vars.items() if (not name.startswith("__")) and (not isfunction(value))}


def compile_type(hagl_type, template_values):

    hagl_type = try_as_syntax_shugar(hagl_type)
    if is_base_hagl_type(hagl_type):
        return hagl_type.gym_type(template_values)

    type_vars = allowed_vars(hagl_type)
    compiled_type = gymnasium.spaces.Dict()

    for field_name in type_vars:
        field_value = type_vars[field_name]
        compiled_type[field_name] = compile_type(field_value, template_values)

    return compiled_type


def compile(observation, action, template_values):

    hagl_template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
    hagl_template_values.update(template_values)

    compiled_observation = compile_type(observation, hagl_template_values)
    compiled_action = compile_type(action, hagl_template_values)

    return compiled_observation, compiled_action


def _construct(hagl_type, gym_dict_value, template_values):

    hagl_type = try_as_syntax_shugar(hagl_type)
    if is_base_hagl_type(hagl_type):
        return hagl_type.construct(gym_dict_value, template_values)

    type_vars = allowed_vars(hagl_type)
    constructed_value = hagl_type()

    for field_name in type_vars:
        field_type = type_vars[field_name]
        constructed_field_value = _construct(field_type, gym_dict_value[field_name], template_values)
        setattr(constructed_value, field_name, constructed_field_value)

    return constructed_value


def _deconstruct(hagl_type, hagl_value, template_values):

    hagl_type = try_as_syntax_shugar(hagl_type)
    if is_base_hagl_type(hagl_type):
        return hagl_type.deconstruct(hagl_value, template_values)

    type_vars = allowed_vars(hagl_type)
    deconstructed_value = OrderedDict()

    for field_name in type_vars:
        field_type = type_vars[field_name]
        deconstructed_field_value = _deconstruct(field_type, getattr(hagl_value, field_name), template_values)
        deconstructed_value[field_name] = deconstructed_field_value

    return deconstructed_value

def construct(hagl_type, gym_dict_value, template_values):

    hagl_template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
    hagl_template_values.update(template_values)
    return _construct(hagl_type, gym_dict_value, hagl_template_values)


def deconstruct(hagl_type, gym_dict_value, template_values):

    hagl_template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
    hagl_template_values.update(template_values)
    return _deconstruct(hagl_type, gym_dict_value, hagl_template_values)