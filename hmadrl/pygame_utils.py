import pygame
from pettingzoo.utils import BaseWrapper


class PyGameFPSWrapper(BaseWrapper):

    def __init__(self, env, fps):
        super().__init__(env)
        self.clock = pygame.time.Clock()
        self.fps = fps

        self.metadata = {"is_parallelizable": True}

    def render(self):
        super().render()
        if pygame.get_init():
            self.clock.tick(self.fps)


class PyGamePolicy:

    def __init__(self, key_action_fn):
        self.key_action_fn = key_action_fn

    def collect_action(self, obs):
        key = pygame.NOEVENT
        if pygame.get_init():
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    key = event.key
        return self.key_action_fn(key, obs)

    def __call__(self, obs):
        return self.collect_action(obs)
