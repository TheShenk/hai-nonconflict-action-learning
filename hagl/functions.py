import hagl
from hagl import HAGLType, get_template
from hagl.base_functions import compile_type


class Function(HAGLType):

    def __init__(self, inner_type, construct_func, deconstruct_func):
        self.inner_type = inner_type
        self.construct_func = construct_func
        self.deconstruct_func = deconstruct_func

    def gym_type(self, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        return compile_type(t_inner_type, template_values)

    def construct(self, gym_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_construct_func = get_template(self.construct_func, template_values)

        return hagl.construct(t_inner_type, t_construct_func(gym_value), template_values)

    def deconstruct(self, hagl_value, template_values):
        t_inner_type = get_template(self.inner_type, template_values)
        t_deconstruct_func = get_template(self.deconstruct_func, template_values)

        return hagl.deconstruct(t_inner_type, t_deconstruct_func(hagl_value), template_values)


class Shift(Function):

    def __init__(self, inner_type, by):
        super().__init__(inner_type, self.construct_func, self.deconstruct_func)
        self.shift_by = by

    def construct_func(self, gym_value):
        return gym_value - self.shift_by

    def deconstruct_func(self, hagl_value):
        return hagl_value + self.shift_by


class Scale(Function):

    def __init__(self, inner_type, at):
        super().__init__(inner_type, self.construct_func, self.deconstruct_func)
        self.scale_at = at

    def construct_func(self, gym_value):
        return gym_value / self.scale_at

    def deconstruct_func(self, hagl_value):
        return hagl_value * self.scale_at


class Normalize(Function):

    def __init__(self, inner_type, left, right):
        super().__init__(inner_type, self.construct_func, self.deconstruct_func)
        self.left = left
        self.right = right
        self.average = (right + left) / 2
        self.range = (right - left) / 2

    def construct_func(self, gym_value):
        return gym_value * self.range + self.average

    def deconstruct_func(self, hagl_value):
        return (hagl_value - self.average) / self.range
