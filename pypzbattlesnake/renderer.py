import pygame


class BattleSnakeRenderer:

    def __init__(self, env):
        self.env = env
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.render()

    def render(self):
        self.screen.fill((128, 128, 128))

    def close(self):
        pygame.exit()
