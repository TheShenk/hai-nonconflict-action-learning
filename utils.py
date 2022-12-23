import random
from abc import abstractmethod
from typing import List, Union, Optional, Tuple

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

# Здесь представлены функции для объединения действий от разных агентов в общее
# Самый простой - для случая, когда не используются векторные среды. В таком случае для всех видов пространств
# действий можем просто соединить действия
NON_VEC_COMBINER = lambda acts: np.concatenate(acts)
# Обработка случая, когда модели используют дискретное пространство действий. В таком случае actions - уже
# массив чисел, который нужно вернуть. Concatenate для него вызовет ошибку.
NON_VEC_DISCRETE_COMBINER = lambda acts: acts
# Для случая, когда используется непрерывное пространство. Необходимо перевести (Число агентов, Число сред, 1, y)
# в (Число сред, Число Агентов, y)
BOX_COMBINER = lambda acts: np.concatenate(acts, axis=1)
# Для случая дискретных пространств. Переводит (Число агентов, Число сред, 1) в (Число сред, Число агентов, 1)
DISCRETE_COMBINER = lambda acts: np.array(acts).transpose()
#TODO: Понять как работает мульти-дискретный случай
MULTI_DISCRETE_COMBINER = lambda acts: np.concatenate(acts, axis=1)

class MultiModelAgent(BaseAgent):
    def __init__(self, models: List[Union[BaseAlgorithm, BaseAgent]], actions_combiner=NON_VEC_COMBINER):
        self.models = models
        self.actions_combiner = actions_combiner

    def predict(self, *args, **kwargs):
        actions = np.array([model.predict(*args, **kwargs)[0] for model in self.models])
        return self.actions_combiner(actions), None


class RandomMultiModelAgent(BaseAgent):

    def __init__(self, models: List[Union[BaseAlgorithm, BaseAgent]]):
        self.models = models
        self.current_model = random.choice(models)

    def predict(self, observation: np.ndarray, state: Optional[Tuple[np.ndarray, ...]] = None,
                episode_start: Optional[np.ndarray] = None, deterministic: bool = False):
        if episode_start.all():
            self.current_model = random.choice(self.models)
        return self.current_model.predict(observation, state, episode_start, deterministic)


def plot_eval_results(eval_log_dir: str):
    data = np.load(f'{eval_log_dir}/evaluations.npz')
    average_results = np.average(data['results'], axis=1)
    plt.figure("Evaluations", figsize=(8, 2))
    plt.plot(data['timesteps'], average_results)
    plt.ylabel("Episode Rewards")
    plt.xlabel("Evaluation timesteps")
    plt.ticklabel_format(style='plain')
    plt.tight_layout()