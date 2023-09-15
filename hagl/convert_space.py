from typing import Any

import gym
import gymnasium
from gymnasium.utils.step_api_compatibility import convert_to_done_step_api


def convert_space(space: gymnasium.Space) -> gym.Space:
    """Converts a gym space to a gymnasium space.

    Args:
        space: the space to convert

    Returns:
        The converted space
    """
    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(n=space.n)
    elif isinstance(space, gymnasium.spaces.Box):
        return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return gym.spaces.MultiDiscrete(nvec=space.nvec)
    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return gym.spaces.MultiBinary(n=space.n)
    elif isinstance(space, gymnasium.spaces.Tuple):
        return gym.spaces.Tuple(spaces=tuple(map(convert_space, space.spaces)))
    elif isinstance(space, gymnasium.spaces.Dict):
        return gym.spaces.Dict(spaces={k: convert_space(v) for k, v in space.spaces.items()})
    elif isinstance(space, gymnasium.spaces.Sequence):
        return gym.spaces.Sequence(space=convert_space(space.feature_space))
    elif isinstance(space, gymnasium.spaces.Graph):
        return gym.spaces.Graph(
            node_space=_convert_space(space.node_space),  # type: ignore
            edge_space=_convert_space(space.edge_space),  # type: ignore
        )
    elif isinstance(space, gymnasium.spaces.Text):
        return gym.spaces.Text(
            max_length=space.max_length,
            min_length=space.min_length,
            charset=space._char_str,
        )
    else:
        return space


class GymnasiumToGym(gym.Env):

    def __init__(self, env: gymnasium.Env):
        self.env = env
        self.action_space = convert_space(self.env.action_space)
        self.observation_space = convert_space(self.env.observation_space)

    def reset(self):
        observation, info = self.env.reset()
        return observation

    def step(self, action):
        step_result = self.env.step(action)
        return convert_to_done_step_api(step_result, False)

    def render(self, mode="human"):
        self.env.render()

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_") and name not in {"_cumulative_rewards"}:
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)