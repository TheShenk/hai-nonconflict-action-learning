from __future__ import annotations

from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType, RenderFrame, ObsType

import hagl

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

        hagl_observation, template_values = self.env.reset()
        self.template_values.update(template_values)

        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)
        return observation

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        gymnasium_action = gymnasium.spaces.unflatten(self.gymnasium_action_space, action)
        hagl_action = hagl.construct(self.hagl_action_space, gymnasium_action, self.template_values)
        #TODO: добавить truncated для поддержки новых версий gymnasium
        hagl_observation, reward, terminated, info, template_values = self.env.step(hagl_action)

        self.template_values.update(template_values)
        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)

        return observation, reward, terminated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return self.env.render()

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



