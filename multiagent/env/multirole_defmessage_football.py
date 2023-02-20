import gym.spaces
import numpy as np

from gym_futbol.envs_v1 import Futbol
from multiagent.action_combiners import NON_VEC_COMBINER


class MultiRoleDefinedMessageFootball(gym.Wrapper):

    def __init__(self, env: Futbol, partner_agent, message, combiner_func = NON_VEC_COMBINER):

        super().__init__(env)
        self.partner_agent = partner_agent
        self.message = message
        self.combiner_func = combiner_func

        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.env.observation_space.shape[0] + 1,))
        if self.env.action_space_type[0] == "box":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4))
        elif self.env.action_space_type[0] == "multi-discrete":
            self.action_space = gym.spaces.Discrete(5)


    def reset(self):
        obs = self.env.reset()
        obs = self._gen_observation_with_message(obs)
        return obs

    def step(self, action):
        partner_action = self.partner_agent.predict(self.env.observation)[0]
        total_action = self.combiner_func([action, partner_action])

        obs, rew, done, info = self.env.step(total_action)
        obs = self._gen_observation_with_message(obs)

        return obs, rew, done, info

    def render(self, mode="human"):
        pass

    def _gen_observation_with_message(self, obs):
        observation = obs.copy()
        observation = np.append(observation, [self.message, ])
        return observation