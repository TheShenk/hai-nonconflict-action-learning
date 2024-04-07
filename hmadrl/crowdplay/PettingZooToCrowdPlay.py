from collections import OrderedDict

import gymnasium
from gymnasium import spaces
import numpy as np


class PettingZooToCrowdPlay:

    def __init__(self, env, use_mask=False):
        self.env = env
        self.unwrapped_env = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env

        self.use_mask = use_mask

        self.action_space = {agent: self.env.action_space(agent) for agent in self.list_of_agents}
        self.observation_space = {agent: gymnasium.spaces.Dict({'obs': self.env.observation_space(agent)})
                                  for agent in self.list_of_agents}
        if self.use_mask:
            for agent in self.list_of_agents:
                self.observation_space[agent]['action_mask'] = gymnasium.spaces.Box(0.0, 1.0,
                                                                                    shape=(self.action_space[agent].n,))
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

    @property
    def list_of_agents(self):
        return self.env.possible_agents

    def reset(self):
        observation, _ = self.env.reset()
        observation = {agent: OrderedDict({"obs": observation[agent]}) for agent in observation}
        if self.use_mask:
            for agent in observation:
                observation[agent]['action_mask'] = np.float32(self.unwrapped_env.action_masks(agent))
        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        observation = {agent: OrderedDict({"obs": observation[agent]}) for agent in observation}
        done = {agent: terminated[agent] or truncated[agent] for agent in terminated}
        if self.use_mask:
            for agent in observation:
                observation[agent]['action_mask'] = np.float32(self.unwrapped_env.action_masks(agent))
        return observation, reward, done, info

    def crowdplay_render(self):
        image = self.env.render()
        return {agent: image for agent in self.list_of_agents}
