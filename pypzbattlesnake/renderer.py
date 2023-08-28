import numpy as np
import pygame
import colorsys

BACKGROUND_COLOR = (255, 255, 255)
CELL_COLOR = (220, 217, 205)
FOOD_COLOR = (255, 20, 147)

RECT_SIZE = 30
FOOD_RADIUS = 10

RECT_PADDING = 5
GRID_PADDING = 10


def random_color():
    h = np.random.default_rng().random()
    return tuple(map(lambda x: int(255*x), colorsys.hls_to_rgb(h, 0.5, 1.0)))


class BattleSnakeRenderer:

    def __init__(self, env):
        self.env = env
        pygame.init()
        pygame.key.set_repeat(1, 1)

        self.screen = pygame.display.set_mode((600, 600))

        self.field: list[list[pygame.Rect]] = []
        self.snake_colors = {snake.color: random_color() for _, snake in self.env.snakes.items()}

        rect_x = GRID_PADDING
        for x in range(self.env.size[0]):
            rect_y = GRID_PADDING
            self.field.append([])
            for y in range(self.env.size[1]):
                self.field[-1].append(pygame.Rect(rect_x, rect_y, RECT_SIZE, RECT_SIZE))
                pygame.draw.rect(self.screen, CELL_COLOR, self.field[-1][-1])
                rect_y += RECT_SIZE + RECT_PADDING
            rect_x += RECT_SIZE + RECT_PADDING

        self.render()

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        self.draw_field()
        self.draw_food()
        for agent, snake in self.env.snakes.items():
            self.draw_snake(snake)

        pygame.display.flip()

    def close(self):
        pygame.exit()

    def draw_snake(self, snake):

        self.draw_head(snake)
        if len(snake.body) != 1:
            self.draw_tail(snake)
            for el in snake.body[1:-1]:
                el_rect = self.field[el[0]][el[1]]
                pygame.draw.rect(self.screen, self.snake_colors[snake.color], el_rect)

    def draw_head(self, snake):
        head_rect = self.field[snake.head()[0]][snake.head()[1]]
        pygame.draw.rect(self.screen, self.snake_colors[snake.color], head_rect)
        pass

    def draw_tail(self, snake):
        tail = snake.body[0]
        tail_rect = self.field[tail[0]][tail[1]]
        pygame.draw.rect(self.screen, self.snake_colors[snake.color], tail_rect)
        pass

    def draw_field(self):
        for row in self.field:
            for cell in row:
                pygame.draw.rect(self.screen, CELL_COLOR, cell)

    def draw_food(self):
        for food in self.env.food:
            food_rect = self.rect_at(food)
            pygame.draw.circle(self.screen, FOOD_COLOR, food_rect.center, FOOD_RADIUS)

    def rect_at(self, pos):
        return self.field[pos[0]][pos[1]]
