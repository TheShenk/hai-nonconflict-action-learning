import gymnasium
import numpy as np

from .base_types import HAGLType
from .base_functions import compile_type, construct, deconstruct
from .template import get_template


class Array(HAGLType):

    def __init__(self, inner_type, elements_count):
        self.inner_type = inner_type
        self.elements_count = elements_count

    def gym_type(self, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_elements_count = get_template(self.elements_count, template_values)

        assert t_elements_count != 0, "Array must have at least 1 element"

        inner_gym_type = compile_type(t_inner_type, template_values)
        return gymnasium.spaces.Tuple([inner_gym_type, ] * t_elements_count)

    def construct(self, gym_value: gymnasium.spaces.Tuple, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_elements_count = get_template(self.elements_count, template_values)

        assert t_elements_count == len(gym_value), ("Got Gymnasium array with number of elements different from the "
                                                    "specified one")

        value = [construct(t_inner_type, t_value, template_values) for t_value in gym_value]
        return value

    def deconstruct(self, hagl_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_elements_count = get_template(self.elements_count, template_values)

        assert t_elements_count == len(hagl_value), ("Got HAGL array with number of elements different from the "
                                                     "specified one")

        value = map(lambda v: deconstruct(t_inner_type, v, template_values), hagl_value)
        return tuple(value)


class EnableIf(HAGLType):

    def __init__(self, inner_type, enabled):
        self.inner_type = inner_type
        self.enabled = enabled

    def gym_type(self, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_enabled = get_template(self.enabled, template_values)

        if t_enabled:
            return compile_type(t_inner_type, template_values)
        else:
            return gymnasium.spaces.Box(shape=(0,), low=np.array([]), high=np.array([]))

    def construct(self, gym_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_enabled = get_template(self.enabled, template_values)

        if t_enabled:
            return construct(t_inner_type, gym_value, template_values)
        else:
            return None

    def deconstruct(self, hagl_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_enabled = get_template(self.enabled, template_values)

        if t_enabled:
            return deconstruct(t_inner_type, hagl_value, template_values)
        else:
            return np.array([])
