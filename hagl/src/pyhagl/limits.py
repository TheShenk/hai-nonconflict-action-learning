import gymnasium
import numpy as np

from .base_types import HAGLType
from .base_functions import compile_type, construct, deconstruct
from .template import get_template
from .exceptions import LimitViolation


class Limit(HAGLType):

    def __init__(self, inner_type, low=None, high=None, equal=None, func=None):
        self.inner_type = inner_type

        self.low = low
        self.high = high
        self.equal = equal
        self.func = func

        self.t_low = None
        self.t_high = None
        self.t_equal = None
        self.t_func = None

        self.range_limit = None
        self.equal_limit = None
        self.func_limit = None

    def gym_type(self, template_values):
        t_inner_type = get_template(self.inner_type, template_values)

        self.t_low = get_template(self.low, template_values)
        self.t_high = get_template(self.high, template_values)
        self.t_equal = get_template(self.equal, template_values)
        self.t_func = get_template(self.func, template_values)

        self.range_limit = (self.t_low is not None) and (self.t_high is not None)
        self.equal_limit = self.t_equal is not None
        self.func_limit = self.t_func is not None

        inner_gym_type = compile_type(t_inner_type, template_values)

        if isinstance(inner_gym_type, gymnasium.spaces.Box):
            if self.range_limit:
                return gymnasium.spaces.Box(self.t_low, self.t_high, inner_gym_type.shape)
            if self.equal_limit:
                return gymnasium.spaces.Box(self.t_equal, self.t_equal, inner_gym_type.shape)
        return inner_gym_type

    def _range_limit_check(self, gym_value):
        if self.range_limit:
            if isinstance(gym_value, np.ndarray):
                limit_correct = np.all(self.t_low <= gym_value) and np.all(gym_value < self.t_high)
            else:
                limit_correct = (self.t_low <= gym_value) and (gym_value < self.t_high)
            if not limit_correct:
                raise LimitViolation(
                    f"Range limit violation. {gym_value} is out of [{self.t_low}, {self.t_high})")

    def _equal_limit_check(self, gym_value):
        if self.equal_limit:
            if isinstance(gym_value, np.ndarray):
                limit_correct = np.all(gym_value == self.t_equal)
            else:
                limit_correct = (gym_value == self.t_equal)
            if not limit_correct:
                raise LimitViolation(f"Equal limit violation. {gym_value} != {self.t_equal}")

    def _func_limit_check(self, gym_value):
        if self.func_limit:
            limit_correct = self.t_func(gym_value)
            if not limit_correct:
                raise LimitViolation(f"Functional limit violation for {gym_value}")

    def construct(self, gym_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)

        self._range_limit_check(gym_value)
        self._equal_limit_check(gym_value)
        self._func_limit_check(gym_value)

        return construct(t_inner_type, gym_value, template_values)

    def deconstruct(self, hagl_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        return deconstruct(t_inner_type, hagl_value, template_values)
