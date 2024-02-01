import pathlib

import numpy as np
import pygame
import colorsys

from .common import Action, ACTION_BY_SHIFT, BASE_COLORS_COUNT

BACKGROUND_COLOR = (255, 255, 255)
CELL_COLOR = (220, 217, 205)
FOOD_COLOR = (255, 20, 147)

RECT_SIZE = 30
FOOD_RADIUS = 10

RECT_PADDING = 5
GRID_PADDING = 10

ROTATE_BY_DIRECTION = {
    Action.RIGHT: lambda x: x,
    Action.LEFT: lambda x: pygame.transform.flip(x, True, False),
    Action.UP: lambda x: pygame.transform.rotate(x, 90),
    Action.DOWN: lambda x: pygame.transform.rotate(x, -90),
    Action.NONE: lambda x: x
}


def random_color():
    h = np.random.default_rng().random()
    return tuple(map(lambda x: int(255 * x), colorsys.hls_to_rgb(h, 0.4, 0.8)))


PALITE = [(0, 104, 139), (0, 104, 139), (159, 0, 255), (159, 0, 255), (212, 137, 28), (212, 137, 28), (0, 155, 92),
          (0, 155, 92)]


def get_shift(cell_a, cell_b):
    return cell_a[0] - cell_b[0], cell_a[1] - cell_b[1]


class BattleSnakeRenderer:

    def __init__(self, env, human_mode=True):
        self.env = env
        self.human_mode = human_mode

        if self.human_mode and not pygame.get_init():
            pygame.init()
            pygame.key.set_repeat(1, 1)

        file_path = pathlib.Path(__file__).parent.resolve()

        size_x = self.env.size[0] * (RECT_SIZE + RECT_PADDING) + GRID_PADDING * 2 - RECT_PADDING
        size_y = self.env.size[1] * (RECT_SIZE + RECT_PADDING) + GRID_PADDING * 2 - RECT_PADDING
        self.screen = pygame.display.set_mode((size_x, size_y))

        self.field: list[list[pygame.Rect]] = []
        self.snake_colors = {snake.color: random_color() if snake.color - BASE_COLORS_COUNT > len(PALITE)
                             else PALITE[snake.color - BASE_COLORS_COUNT] for _, snake in self.env.snakes.items()}

        rect_x = GRID_PADDING
        for x in range(self.env.size[0]):
            rect_y = GRID_PADDING
            self.field.append([])
            for y in range(self.env.size[1]):
                self.field[-1].append(pygame.Rect(rect_x, rect_y, RECT_SIZE, RECT_SIZE))
                pygame.draw.rect(self.screen, CELL_COLOR, self.field[-1][-1])
                rect_y += RECT_SIZE + RECT_PADDING
            rect_x += RECT_SIZE + RECT_PADDING

        self.head_img = pygame.image.load(file_path / 'resources' / 'head.svg').convert_alpha()
        self.head_img = pygame.transform.scale(self.head_img, (RECT_SIZE, RECT_SIZE))
        for _, snake in self.env.snakes.items():
            snake.render_head = self.head_img.copy()
            snake.render_head.fill(self.snake_colors[snake.color], special_flags=pygame.BLEND_RGB_MAX)

        self.tail_img = pygame.image.load(file_path / 'resources' / 'tail.svg').convert_alpha()
        self.tail_img = pygame.transform.scale(self.tail_img, (RECT_SIZE, RECT_SIZE))
        for _, snake in self.env.snakes.items():
            snake.render_tail = self.tail_img.copy()
            snake.render_tail.fill(self.snake_colors[snake.color], special_flags=pygame.BLEND_RGB_MAX)

    def render(self):
        self.screen.fill(BACKGROUND_COLOR)

        self.draw_field()
        self.draw_food()
        for agent, snake in self.env.snakes.items():
            self.draw_snake(snake)

        if self.human_mode:
            pygame.display.flip()
        else:
            return pygame.surfarray.array3d(self.screen)

    def close(self):
        pygame.quit()

    def draw_snake(self, snake):

        self.draw_head(snake, snake.action)
        if len(snake.body) != 1:
            shift = get_shift(snake.body[0], snake.body[1])
            direction = ACTION_BY_SHIFT[shift]
            self.draw_tail(snake, direction)
            for el in snake.body[1:-1]:
                el_rect = self.field[el[0]][el[1]]
                pygame.draw.rect(self.screen, self.snake_colors[snake.color], el_rect)

    def draw_head(self, snake, direction):
        head_rect = self.field[snake.head()[0]][snake.head()[1]]
        head = ROTATE_BY_DIRECTION[direction](snake.render_head)
        self.screen.blit(head, head_rect.topleft)

    def draw_tail(self, snake, direction):
        tail_rect = self.field[snake.tail()[0]][snake.tail()[1]]
        tail = ROTATE_BY_DIRECTION[direction](snake.render_tail)
        self.screen.blit(tail, tail_rect.topleft)

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
