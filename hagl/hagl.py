from abc import abstractmethod

import gymnasium

DIMENSIONS_COUNT = 2

class HAGLType:
    pass

class Velocity(HAGLType):

    @staticmethod
    def gym_type():
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(DIMENSIONS_COUNT,))

class Position(HAGLType):

    @staticmethod
    def gym_type():
        return gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(DIMENSIONS_COUNT,))

class Array(HAGLType):

    def __init__(self, inner_type, elements_count):
        self.inner_type = inner_type
        self.inner_gym_type = compile_type(inner_type)
        self.elements_count = elements_count

    def gym_type(self):
        return gymnasium.spaces.Tuple([self.inner_gym_type,] * self.elements_count)

def set_dimension(dimensions_count):
    global DIMENSIONS_COUNT
    DIMENSIONS_COUNT = dimensions_count

def compile_type(hagl_type):

    if hasattr(hagl_type, "gym_type"):
        return hagl_type.gym_type()

    type_vars = vars(hagl_type)
    compiled_type = gymnasium.spaces.Dict()

    for field_name in type_vars:
        if not field_name.startswith("__"):
            field_value = type_vars[field_name]
            compiled_type[field_name] = compile_type(field_value)

    return compiled_type

def compile(observation, action):

    compiled_observation = compile_type(observation)
    compiled_action = compile_type(action)

    return compiled_observation, compiled_action