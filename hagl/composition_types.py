import gym.spaces
import gymnasium

import hagl
from hagl import HAGLType, get_template
from hagl.base_functions import compile_type


class Array(HAGLType):

    def __init__(self, inner_type, elements_count):
        self.inner_type = inner_type
        self.elements_count = elements_count

    def gym_type(self, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_elements_count = get_template(self.elements_count, template_values)

        inner_gym_type = compile_type(t_inner_type, template_values)
        return gymnasium.spaces.Tuple([inner_gym_type,] * t_elements_count)

    def construct(self, gym_value: gym.spaces.Tuple, template_values):
        t_inner_type = get_template(self.inner_type, template_values)

        value = [hagl.construct(t_inner_type, t_value, template_values) for t_value in gym_value]
        return value

    def deconstruct(self, hagl_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)

        value = map(lambda v: hagl.deconstruct(t_inner_type, v, template_values), hagl_value)
        return tuple(value)