import enum

EMPTY_COLOR = 0
FOOD_COLOR = 1
BASE_COLORS_COUNT = 2  # empty, food


class Action(enum.IntEnum):
    UP = enum.auto()
    DOWN = enum.auto()
    LEFT = enum.auto()
    RIGHT = enum.auto()
    NONE = enum.auto()


SHIFT_BY_ACTION = {
    Action.UP: [0, -1],
    Action.DOWN: [0, 1],
    Action.LEFT: [-1, 0],
    Action.RIGHT: [1, 0],
    Action.NONE: [0, 0]
}

OPPOSITE_ACTION = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT
}


ACTION_BY_SHIFT = dict((tuple(v), k) for k, v in SHIFT_BY_ACTION.items())
