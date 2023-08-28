from __future__ import annotations

import functools
from typing import SupportsFloat, Any, Callable

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType, RenderFrame, ObsType
import pettingzoo

import hagl
from hagl.convert_space import convert_space


class HAGLWrapper(gymnasium.Env):

    def __init__(self, env, template_values=None):
        super().__init__()

        if template_values is None:
            template_values = dict()

        self.env = env
        self.hagl_action_space, self.hagl_observation_space = self.env.action_space, self.env.observation_space
        self.template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
        self.template_values.update(template_values)

        self.gymnasium_action_space, self.gymnasium_observation_space \
            = hagl.compile(self.hagl_action_space, self.hagl_observation_space, template_values)
        self.action_space = gymnasium.spaces.flatten_space(self.gymnasium_action_space)
        self.observation_space = gymnasium.spaces.flatten_space(self.gymnasium_observation_space)

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed,
                      options=options)

        hagl_observation, template_values, info = self.env.reset()
        self.template_values.update(template_values)

        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)
        return observation, info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        gymnasium_action = gymnasium.spaces.unflatten(self.gymnasium_action_space, action)
        hagl_action = hagl.construct(self.hagl_action_space, gymnasium_action, self.template_values)
        hagl_observation, reward, terminated, truncated, info, template_values = self.env.step(hagl_action)

        self.template_values.update(template_values)
        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)

        return observation, reward, terminated, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()


def for_each_agent(agents: dict, func: Callable[[Any], Any]):
    func_agents = {}
    for agent, data in agents.items():
        func_agents[agent] = func(agent, data)
    return func_agents


class HAGLParallelWrapper(pettingzoo.ParallelEnv):

    def __init__(self, env, template_values=None):
        super().__init__()

        if template_values is None:
            template_values = dict()

        self.env = env
        self.template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
        self.template_values.update(template_values)

        if self.env.init_template:
            self.template_values.update(self.env.init_template())

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def _process_observation(self, agent_id, hagl_observation):
        hagl_observation_space = self.env.observation_space(agent_id)
        gymnasium_observation_space = hagl.compile_one(hagl_observation_space, self.template_values)
        gymnasium_observation = hagl.deconstruct(hagl_observation_space, hagl_observation, self.template_values)
        return gymnasium.spaces.flatten(gymnasium_observation_space, gymnasium_observation)

    def _process_action(self, agent_id, action):
        hagl_action_space = self.env.action_space(agent_id)
        gymnasium_action_space = hagl.compile_one(hagl_action_space, self.template_values)
        gymnasium_action = gymnasium.spaces.unflatten(gymnasium_action_space, action)
        return hagl.construct(hagl_action_space, gymnasium_action, self.template_values)

    def reset(self, seed: int | None = None, options: dict | None = None):

        hagl_observation, info, template_values = self.env.reset(seed, options)
        self.template_values.update(template_values)
        observation = for_each_agent(hagl_observation, self._process_observation)

        return observation, info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        hagl_action = for_each_agent(action, self._process_action)
        hagl_observation, reward, terminated, truncated, info, template_values = self.env.step(hagl_action)
        observation = for_each_agent(hagl_observation, self._process_observation)

        return observation, reward, terminated, truncated, info

    def render(self, mode: str = "human") -> RenderFrame | list[RenderFrame] | None:
        return self.env.render(mode)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        hagl_action_space = self.env.action_space(agent)
        gymnasium_action_space = hagl.compile_one(hagl_action_space, self.template_values)
        return gymnasium.spaces.flatten_space(gymnasium_action_space)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        hagl_observation_space = self.env.observation_space(agent)
        gymnasium_observation_space = hagl.compile_one(hagl_observation_space, self.template_values)
        return gymnasium.spaces.flatten_space(gymnasium_observation_space)


class HAGLAECWrapper(pettingzoo.AECEnv):

    def __init__(self, env, template_values=None):
        super().__init__()

        if template_values is None:
            template_values = dict()

        self.env = env
        self.template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
        self.template_values.update(template_values)

    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_") and name not in {"_cumulative_rewards"}:
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self.env, name)

    def _process_observation(self, agent_id, hagl_observation):
        hagl_observation_space = self.env.observation_space(agent_id)
        gymnasium_observation_space = hagl.compile_one(hagl_observation_space, self.template_values)
        gymnasium_observation = hagl.deconstruct(hagl_observation_space, hagl_observation, self.template_values)
        return gymnasium.spaces.flatten(gymnasium_observation_space, gymnasium_observation)

    def _process_action(self, agent_id, action):
        hagl_action_space = self.env.action_space(agent_id)
        gymnasium_action_space = hagl.compile_one(hagl_action_space, self.template_values)
        gymnasium_action = gymnasium.spaces.unflatten(gymnasium_action_space, action)
        return hagl.construct(hagl_action_space, gymnasium_action, self.template_values)

    def reset(self):
        self.env.reset()

    def step(self, action):
        self.env.step(self._process_action(self.env.agent_selection, action))

    def observe(self, agent):
        return self._process_observation(agent, self.env.observe(agent))

    def render(self, mode: str = "human") -> RenderFrame | list[RenderFrame] | None:
        return self.env.render(mode)

    def action_space(self, agent):
        hagl_action_space = self.env.action_space(agent)
        gymnasium_action_space = hagl.compile_one(hagl_action_space, self.template_values)
        return convert_space(gymnasium.spaces.flatten_space(gymnasium_action_space))

    def observation_space(self, agent):
        hagl_observation_space = self.env.observation_space(agent)
        gymnasium_observation_space = hagl.compile_one(hagl_observation_space, self.template_values)
        return convert_space(gymnasium.spaces.flatten_space(gymnasium_observation_space))

    def close(self):
        self.env.close()


class HAGLModel:

    def __init__(self, model, hagl_action_space, hagl_observation_space, template_values=None):
        if template_values is None:
            template_values = dict()
        self.model = model

        self.hagl_action_space, self.hagl_observation_space = hagl_action_space, hagl_observation_space
        self.template_values = hagl.template.DEFAULT_TEMPLATE_VALUES.copy()
        self.template_values.update(template_values)

        self.gymnasium_action_space, self.gymnasium_observation_space \
            = hagl.compile(self.hagl_action_space, self.hagl_observation_space, template_values)
        self.action_space = gymnasium.spaces.flatten_space(self.gymnasium_action_space)
        self.observation_space = gymnasium.spaces.flatten_space(self.gymnasium_observation_space)

    def predict(self, hagl_observation):
        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)

        action, _ = self.model.predict(observation)

        gymnasium_action = gymnasium.spaces.unflatten(self.gymnasium_action_space, action)
        hagl_action = hagl.construct(self.hagl_action_space, gymnasium_action, self.template_values)

        return hagl_action
