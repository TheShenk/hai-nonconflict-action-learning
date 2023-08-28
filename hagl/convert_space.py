import gym
import gymnasium


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
