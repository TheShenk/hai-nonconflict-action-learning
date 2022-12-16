from abc import abstractmethod
from typing import List, Union

import gym
import numpy as np
import pymunk
import pymunk.pygame_util
import pymunk.matplotlib_util

import pygame
from IPython import display
import matplotlib.pyplot as plt
from gym import Wrapper
from stable_baselines3.common.base_class import BaseAlgorithm

from agents.base_agent import BaseAgent


class BaseVisualizer:

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def visualize(self, reward):
        pass

    def run(self, model):
        ob = self.env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _states = model.predict(ob)
            ob, reward, done, info = self.env.step(action)
            self.visualize(reward)
            total_reward += reward
        return total_reward

class NoVisualizer(BaseVisualizer):

    def __init__(self, env):
        super().__init__(env)

    def visualize(self, reward):
        pass

class PygameVisualizer(BaseVisualizer):
    def __init__(self, env, res, fps=60):
        super().__init__(env)
        pygame.init()
        self.surface = pygame.display.set_mode(res)
        self.clock = pygame.time.Clock()

        translation = (4, 2)
        scale_factor = min(res[0] / (env.width + translation[0] * 2), res[1] / (env.height + translation[1] * 2))
        self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)
        self.draw_options.transform = pymunk.Transform.scaling(scale_factor) @ pymunk.Transform.translation(
            translation[0], translation[1])
        self.fps = fps

    def visualize(self, reward):
        self.surface.fill("black")
        self.env.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(self.fps)

    @staticmethod
    def close():
        pygame.quit()


class MatplotlibVisualizer(BaseVisualizer):

    def __init__(self, env):
        super().__init__(env)

    def visualize(self, reward):
        padding = 5

        plt.clf()
        ax = plt.axes(
            xlim=(0 - padding, self.env.width + padding),
            ylim=(0 - padding, self.env.height + padding)
        )
        ax.set_aspect("equal")

        draw_options = pymunk.matplotlib_util.DrawOptions(ax)
        self.env.space.debug_draw(draw_options)
        plt.title(f"Reward: {reward}", loc='left')
        display.display(plt.gcf())
        display.clear_output(wait=True)


class ConstantEnv(gym.Env):

    def __init__(self, observation, reward, done, info):
        self.observation = observation
        self.reward = reward
        self.done = done
        self.info = info

    def step(self, action):
        return self.reset()

    def render(self, mode="human"):
        pass

    def reset(self):
        return self.observation, self.reward, self.done, self.info


class TransformAction(Wrapper):

    def __init__(self, env, space):
        super().__init__(env)
        self.action_space = space


class TestStaticAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def predict(self, *args, **kwargs):
        return np.array([[-1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]), None


class MultiModelAgent:
    def __init__(self, models: List[Union[BaseAlgorithm, BaseAgent]]):
        self.models = models

    def predict(self, *args, **kwargs):
        actions = np.array([model.predict(*args, **kwargs)[0] for model in self.models])

        if actions.ndim == 1:
            return actions, None
        return np.concatenate(actions), None
