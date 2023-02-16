import gym

from gym_futbol.envs_v1 import Futbol
from multiagent.action_combiners import NON_VEC_COMBINER


class MultiRoleFootball(gym.Env):

    def __init__(self, env: Futbol, partner_agent, combiner_func = NON_VEC_COMBINER):
        self.env = env
        self.partner_agent = partner_agent
        self.combiner_func = combiner_func

        self.observation_space = self.env.observation_space

        if self.env.action_space_type[0] == "box":
            self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1, 4))
        elif self.env.action_space_type[0] == "multi-discrete":
            self.action_space = gym.spaces.Discrete(5)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        partner_action = self.partner_agent.predict(self.env.observation)[0]
        total_action = self.combiner_func([action, partner_action])
        return self.env.step(total_action)

    def render(self, mode="human"):
        pass