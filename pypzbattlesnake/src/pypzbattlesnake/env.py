import functools
import math
from typing import Tuple, Dict, Set

import gymnasium
import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID

from .common import Action, SHIFT_BY_ACTION, OPPOSITE_ACTION, BASE_COLORS_COUNT, FOOD_COLOR, EMPTY_COLOR
from .renderer import BattleSnakeRenderer

ActionSpace = gymnasium.spaces.Discrete(5)


class Snake:

    _color = FOOD_COLOR

    def __init__(self, init_pos, team):

        self.body: list[tuple[int, int]] = [init_pos]
        self.team = team

        Snake._color += 1
        self.color = Snake._color
        Snake._color += 1
        self.head_color = Snake._color
        self.action = Action.NONE
        self.health = 100

    def step(self, action):

        if action != Action.NONE and self.action != OPPOSITE_ACTION[action]:
            self.action = action

        self.health -= 1

        shift = SHIFT_BY_ACTION[self.action]

        head_block = self.body[-1]
        next_block = (head_block[0] + shift[0], head_block[1] + shift[1])

        self.body.append(next_block)

    def heal(self):
        self.health = 100

    def head(self):
        return self.body[-1]

    def tail(self):
        return self.body[0]


class BattleSnake(pettingzoo.ParallelEnv):

    def __init__(self, teams_count,
                 team_snakes_count, size: Tuple[int, int] = (15, 15),
                 min_food_count: int = 5, food_spawn_interval=5, food_reward=0.1, render_mode="human"):

        super().__init__()
        self.teams_count = teams_count
        self.team_snakes_count = team_snakes_count
        self.snakes_count = teams_count * team_snakes_count
        self.colors_count = 2 * self.snakes_count + BASE_COLORS_COUNT

        self.render_mode = render_mode

        self.size = size
        self.max_dist = sum(size)
        self.init_positions = self._get_init_pos()

        self.min_food_count = min_food_count
        self.food_spawn_interval = food_spawn_interval
        self.food_reward = food_reward
        self.food_spawn_timer = 0

        self.reward = {}
        self.food: Set[Tuple[int, int]] = set()
        self.snakes: Dict[str, Snake] = {}
        self.eliminated_agents: set[str] = set()

        self.rng = np.random.default_rng()

        self.teams = [[f"snake_{team_id}_{snake_id}" for snake_id in range(self.team_snakes_count)]
                      for team_id in range(self.teams_count)]
        self.possible_agents = sum(self.teams, [])

        self.action_spaces = {agent: Action for agent in self.possible_agents}

        self.color_mapping = None
        self.renderer = None
        self.metadata = {"render.mode": ["human"]}

    def _get_init_pos(self):
        inner_square_size = (self.size[0] - 2, self.size[1] - 2)
        place_cells_count = 2 * (sum(inner_square_size) - 2)
        directions = [[1, 0], [0, 1], [-1, 0], [0, -1]]

        current_pos = (0, 0)
        current_direction = 0

        possible_pos = [current_pos]
        for i in range(0, place_cells_count):
            current_pos = (current_pos[0] + directions[current_direction][0],
                           current_pos[1] + directions[current_direction][1])
            if current_pos in {(inner_square_size[0] - 1, 0), (0, inner_square_size[1] - 1),
                               (inner_square_size[0] - 1, inner_square_size[1] - 1)}:
                current_direction += 1
            possible_pos.append(current_pos)

        place_gap = place_cells_count // self.snakes_count
        return [(possible_pos[i][0] + 1, possible_pos[i][1] + 1) for i in range(0, place_cells_count, place_gap)]

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.rng = np.random.default_rng()
        self.renderer = None

        self.agents = self.possible_agents.copy()

        Snake._color = FOOD_COLOR
        self.snakes = {}
        snake_position = iter(self.init_positions)
        for idx, team in enumerate(self.teams):
            for agent in team:
                self.snakes[agent] = Snake(next(snake_position), team=idx)

        self.color_mapping = self.get_color_mapping()
        self.food = set()
        for _ in range(self.min_food_count):
            self.food.add(self.random_empty_position())
        self.food_spawn_timer = 0

        self.reward = {agent: 0.0 for agent in self.agents}
        self.eliminated_agents = set()

        observation = {agent: self.get_observation(agent) for agent in self.agents}
        info = {agent: {} for agent in self.agents}

        return observation, info

    def step(self, action: Dict[str, int]):

        action = {agent: Action(act + 1) for agent, act in action.items()}

        before_food_distance = {agent: self.find_closest_food(self.snakes[agent].head())[1] for agent in self.snakes}

        for agent, act in action.items():
            if agent in self.snakes:
                self.snakes[agent].step(act)

        after_food_distance = {agent: self.find_closest_food(self.snakes[agent].head())[1] for agent in self.snakes}
        self.reward = {
            agent: (before_food_distance[agent] - after_food_distance[agent]) / self.max_dist * self.food_reward
            if agent in self.snakes else 0 for agent in action}

        self.check_health()
        self.check_border_collisions()
        self.check_food_collisions()
        self.check_self_collisions()
        self.check_head_head_collisions()
        self.check_another_collisions()

        self.spawn_food()

        end_game = self.check_end_game()

        observation = {agent: self.get_observation(agent) for agent in action}
        if end_game:
            terminated = {agent: True for agent in action}
            self.eliminate(self.agents.copy())
        else:
            terminated = {agent: agent in self.eliminated_agents for agent in action}
        truncated = {agent: False for agent in action}
        info = {agent: {} for agent in action}

        return observation, self.reward, terminated, truncated, info

    def find_closest_food(self, pos):

        def distance(point1, point2):
            return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

        closest_food = next(iter(self.food))
        min_dist = distance(closest_food, pos)

        for food in self.food:
            dist = distance(food, pos)
            if dist < min_dist:
                closest_food = food
                min_dist = dist

        return closest_food, min_dist

    def check_health(self):
        eliminate_agents = []
        for agent in self.agents:
            if self.snakes[agent].health <= 0:
                eliminate_agents.append(agent)
        self.eliminate(eliminate_agents)

    def check_border_collisions(self):
        eliminate_agents = []
        for agent in self.agents:
            head = self.snakes[agent].head()
            if not (0 <= head[0] < self.size[0]) or not (0 <= head[1] < self.size[1]):
                eliminate_agents.append(agent)
        self.eliminate(eliminate_agents)

    def check_food_collisions(self):
        for agent in self.agents:
            snake = self.snakes[agent]
            snake_head = snake.body[-1]
            if snake_head in self.food:
                snake.heal()
                self.food.remove(snake_head)
                self.reward[agent] = self.food_reward
            else:
                snake.body.pop(0)

    def check_self_collisions(self):
        eliminate_agents = []
        for agent in self.agents:
            snake = self.snakes[agent]
            snake_head = snake.head()
            if snake_head in snake.body[:-1]:
                eliminate_agents.append(agent)
        self.eliminate(eliminate_agents)

    def check_head_head_collisions(self):
        eliminate_agents = []
        all_heads = [self.snakes[agent].head() for agent in self.agents]
        for agent in self.agents:
            head = self.snakes[agent].head()
            if all_heads.count(head) != 1:
                eliminate_agents.append(agent)
        self.eliminate(eliminate_agents)

    def check_another_collisions(self):
        eliminate_agents = []
        for agent in self.agents:
            for other_agent in self.agents:
                if agent != other_agent and self.snakes[agent].head() in self.snakes[other_agent].body:
                    eliminate_agents.append(agent)
                    self.reward[other_agent] = -1.0 if self.snakes[agent].team == self.snakes[other_agent].team else 1.0
        self.eliminate(eliminate_agents)

    def eliminate(self, eliminate_agent):
        for agent in eliminate_agent:
            self.reward[agent] = -1.0
            self.agents.remove(agent)
            snake = self.snakes[agent]
            for teammate in self.teams[snake.team]:
                if teammate in self.reward:
                    self.reward[teammate] = -1.0
            self.snakes.pop(agent)
            self.eliminated_agents.add(agent)

    def get_color_mapping(self):

        def agent_color_mapping(snake_id):
            snake = self.snakes[snake_id]
            snake_color_mapping = [0, ] * self.colors_count
            snake_color_mapping[1] = 1
            current_agent_color = BASE_COLORS_COUNT
            snake_color_mapping[snake.color] = current_agent_color
            snake_color_mapping[snake.head_color] = current_agent_color + 1
            for teammate in self.teams[snake.team]:
                if teammate != snake_id:
                    current_agent_color += 2
                    snake_color_mapping[self.snakes[teammate].color] = current_agent_color
                    snake_color_mapping[self.snakes[teammate].head_color] = current_agent_color + 1
            for opponent_team in range(len(self.teams)):
                if opponent_team != snake.team:
                    for opponent in self.teams[opponent_team]:
                        current_agent_color += 2
                        snake_color_mapping[self.snakes[opponent].color] = current_agent_color
                        snake_color_mapping[self.snakes[opponent].head_color] = current_agent_color + 1
            return snake_color_mapping

        return {snake_id: agent_color_mapping(snake_id) for snake_id in self.possible_agents}

    def as_array(self):

        field = np.zeros(self.size, dtype=np.int64)
        for agent, snake in self.snakes.items():
            for x, y in snake.body:
                field[x][y] = snake.color
            head = snake.head()
            field[head[0]][head[1]] = snake.head_color

        for x, y in self.food:
            field[x][y] = FOOD_COLOR

        return field

    def get_observation(self, observation_snake_id):
        observation = self.as_array()
        # for x in range(self.size[0]):
        #     for y in range(self.size[1]):
        #         observation[x][y] = self.color_mapping[observation_snake_id][observation[x][y]]

        observation = observation.flat
        # directions = np.full((self.snakes_count,), Action.NONE, np.int64)
        # for snake_id, snake in self.snakes.items():
        #     directions[self.color_mapping[observation_snake_id][snake.color] - BASE_COLORS_COUNT] = snake.action
        # observation = np.append(observation, directions)
        return observation

    def random_empty_position(self):
        rng = np.random.default_rng()
        field = self.as_array()
        empty_indices = np.argwhere(field == EMPTY_COLOR)
        index = rng.choice(empty_indices, axis=0)
        return tuple(index)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: AgentID):
        dim = math.prod(self.size)
        dim = np.full((dim,), BASE_COLORS_COUNT + 2*self.snakes_count)
        return gymnasium.spaces.MultiDiscrete(dim)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: AgentID):
        return ActionSpace

    def render(self) -> None:

        if self.renderer is None:
            self.renderer = BattleSnakeRenderer(self, self.render_mode == "human")

        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()

    def init_template(self):
        return {"colors_count": self.snakes_count + BASE_COLORS_COUNT,
                "field_x": self.size[0],
                "field_y": self.size[1]}

    def check_end_game(self):
        teams_left = set()
        for name, snake in self.snakes.items():
            teams_left.add(snake.team)

        return len(teams_left) <= 1

    def spawn_food(self):

        while len(self.food) < self.min_food_count:
            self.food.add(self.random_empty_position())

        self.food_spawn_timer += 1
        if self.food_spawn_timer >= self.food_spawn_interval:
            self.food.add(self.random_empty_position())
            self.food_spawn_timer = 0
