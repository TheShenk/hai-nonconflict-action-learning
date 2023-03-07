import gymnasium

class HAGLType:
    pass

def compile_type(hagl_type, template_values):

    if hasattr(hagl_type, "gym_type"):
        return hagl_type.gym_type(template_values)

    type_vars = vars(hagl_type)
    compiled_type = gymnasium.spaces.Dict()

    for field_name in type_vars:
        if not field_name.startswith("__"):
            field_value = type_vars[field_name]
            compiled_type[field_name] = compile_type(field_value, template_values)

    return compiled_type

def compile(observation, action, template_values):

    compiled_observation = compile_type(observation, template_values)
    compiled_action = compile_type(action, template_values)

    return compiled_observation, compiled_action