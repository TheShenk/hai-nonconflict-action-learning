import gym
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    def __init__(self, env: gym.Env, action_space=None):
        super().__init__(env)
        self.action_space = action_space

    def predict(self, observation):
        if self.action_space:
            return self.action_space.sample(), None
        return self.env.action_space.sample(), None