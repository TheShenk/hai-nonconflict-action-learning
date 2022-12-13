from abc import abstractmethod
import gym

class BaseAgent:

    def __init__(self, env: gym.Env):
        self.env = env
    @abstractmethod
    def predict(self, observation):
        pass