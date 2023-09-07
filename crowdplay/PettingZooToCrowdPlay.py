from typing import Any

from gymnasium import spaces
import numpy as np


class PettingZooToCrowdPlay:

    def __init__(self, env):
        self.env = env
        self.list_of_agents = self.env.possible_agents

        self.action_space = {agent: self.env.action_space(agent) for agent in self.list_of_agents}
        self.observation_space = {agent: self.env.observation_space(agent) for agent in self.list_of_agents}
        self.num_players = self.env.max_num_agents

    def __getattr__(self, name: str):
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def get_noop_action(self, agent):
        def get_noop_for_space(space):
            if isinstance(space, spaces.Box):
                return np.zeros(space.shape)
            if isinstance(space, spaces.Discrete):
                return 0
            if isinstance(space, spaces.Dict):
                return {key: get_noop_for_space(space.spaces[key]) for key in space}
            if isinstance(space, spaces.MultiDiscrete):
                return np.zeros(space.shape)
            if isinstance(space, spaces.MultiBinary):
                return np.zeros(space.n)
            if isinstance(space, spaces.Tuple):
                return [get_noop_for_space(s) for s in space.spaces]
            return 0

        return get_noop_for_space(self.action_space[agent])
