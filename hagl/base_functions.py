from collections import OrderedDict

import gymnasium

import hagl
from hagl import HAGLType


def is_base_hagl_type(val):
    if type(val) == type:
        return issubclass(val, HAGLType)
    else:
        return issubclass(type(val), HAGLType)


def allowed_vars(hagl_type):
    type_vars = vars(hagl_type)
    return {name: value for name, value in type_vars.items() if not name.startswith("__")}


def compile_type(hagl_type, template_values):
    if is_base_hagl_type(hagl_type):
        return hagl_type.gym_type(template_values)

    elif isinstance(hagl_type, list):
        hagl_type = hagl.composition_types.Array(hagl_type[0], hagl_type[1])
        return hagl_type.gym_type(template_values)

    type_vars = allowed_vars(hagl_type)
    compiled_type = gymnasium.spaces.Dict()

    for field_name in type_vars:
        field_value = type_vars[field_name]
        compiled_type[field_name] = compile_type(field_value, template_values)

    return compiled_type


def compile(observation, action, template_values):
    compiled_observation = compile_type(observation, template_values)
    compiled_action = compile_type(action, template_values)

    return compiled_observation, compiled_action


def construct(hagl_type, gym_dict_value):
    if is_base_hagl_type(hagl_type):
        return hagl_type.construct(gym_dict_value)

    elif isinstance(hagl_type, list):
        hagl_type = hagl.composition_types.Array(hagl_type[0], hagl_type[1])
        return hagl_type.construct(gym_dict_value)

    type_vars = allowed_vars(hagl_type)
    constructed_value = hagl_type()

    for field_name in type_vars:
        field_type = type_vars[field_name]
        constructed_field_value = construct(field_type, gym_dict_value[field_name])
        setattr(constructed_value, field_name, constructed_field_value)

    return constructed_value


def deconstruct(hagl_type, hagl_value):
    if is_base_hagl_type(hagl_type):
        return hagl_type.deconstruct(hagl_value)

    elif isinstance(hagl_type, list):
        hagl_type = hagl.composition_types.Array(hagl_type[0], hagl_type[1])
        return hagl_type.deconstruct(hagl_value)

    type_vars = allowed_vars(hagl_type)
    deconstructed_value = OrderedDict()

    for field_name in type_vars:
        field_type = type_vars[field_name]
        deconstructed_field_value = deconstruct(field_type, getattr(hagl_value, field_name))
        deconstructed_value[field_name] = deconstructed_field_value

    return deconstructed_value