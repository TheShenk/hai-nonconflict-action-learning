from gymnasium.spaces.utils import unflatten, flatten, flatten_space
from .base_types import HAGLType


class GymnasiumStraight:
    pass


@flatten_space.register(GymnasiumStraight)
def _flatten_space_straight(space: GymnasiumStraight) -> GymnasiumStraight:
    return space


@unflatten.register(GymnasiumStraight)
def _unflatten_straight(space: GymnasiumStraight, value):
    return value


@flatten.register(GymnasiumStraight)
def _flatten_straight(space: GymnasiumStraight, value):
    return value


class Straight(HAGLType):

    def gym_type(self, template_values):
        return GymnasiumStraight()

    def construct(self, gym_value, template_values):
        return gym_value

    def deconstruct(self, hagl_value, template_values):
        return hagl_value
