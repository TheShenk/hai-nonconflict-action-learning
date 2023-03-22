from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType

import hagl


class HAGLWrapper(gymnasium.Wrapper):

    def __init__(self, env: gymnasium.Env, action_space, observation_space, template_values):
        super().__init__(env)

        self.hagl_action_space, self.hagl_observation_space = action_space, observation_space
        self.template_values = template_values

        self.gymnasium_action_space, self.gymnasium_observation_space \
            = hagl.compile(self.hagl_action_space, self.hagl_observation_space, template_values)
        self.action_space = gymnasium.spaces.flatten_space(self.gymnasium_action_space)
        self.observation_space = gymnasium.spaces.flatten_space(self.gymnasium_observation_space)
    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        gymnasium_action = gymnasium.spaces.unflatten(self.gymnasium_action_space, action)
        hagl_action = hagl.construct(self.hagl_action_space, gymnasium_action, self.template_values)
        hagl_observation, reward, terminated, truncated, info = self.env.step(hagl_action)
        gymnasium_observation = hagl.deconstruct(self.hagl_observation_space, hagl_observation, self.template_values)
        observation = gymnasium.spaces.flatten(self.gymnasium_observation_space, gymnasium_observation)

        return observation, reward, terminated, truncated, info