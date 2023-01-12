import gym
from stable_baselines3.common.type_aliases import GymStepReturn


class TwoSideRewardMonitor(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.total_left_reward = 0
        self.total_right_reward = 0

    def step(self, action) -> GymStepReturn:
        observation, reward, done, info = self.env.step(action)

        self.total_left_reward += info["left_reward"]
        info["total_left_reward"] = self.total_left_reward

        self.total_right_reward += info["right_reward"]
        info["total_right_reward"] = self.total_right_reward

        if done:
            self.total_left_reward = 0
            self.total_right_reward = 0

        return observation, reward, done, info

