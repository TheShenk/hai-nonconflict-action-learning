import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID

from hagl import T
from hagl.python_types import Integer
from pypzbattlesnake.common import Action, SHIFT_BY_ACTION
from pypzbattlesnake.renderer import BattleSnakeRenderer


class Observation:
    field = [[Integer(T("colors_count")), T("field_x")], T("field_y")]


EMPTY_COLOR = 0
FOOD_COLOR = 1

BASE_COLORS_COUNT = 2  # empty, food


class Snake:

    _color = FOOD_COLOR

    def __init__(self, init_pos, team):

        self.body: list[tuple[int, int]] = [init_pos]
        self.team = team

        Snake._color += 1
        self.color = Snake._color
        self.action = Action.NONE
        self.health = 100

    def step(self, action):

        if action != Action.NONE:
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
                 team_snakes_count, size: tuple[int, int] = (15, 15),
                 min_food_count: int = 5, food_spawn_interval=5, food_reward=0.1):

        super().__init__()
        self.teams_count = teams_count
        self.team_snakes_count = team_snakes_count
        self.snakes_count = teams_count * team_snakes_count

        self.size = size
        self.min_food_count = min_food_count
        self.food_spawn_interval = food_spawn_interval
        self.food_reward = food_reward
        self.food_spawn_timer = 0

        self.reward = {}
        self.food: set[tuple[int, int]] = set()
        self.snakes: dict[str, Snake] = {}
        self.eliminated_agents: set[str] = set()

        self.rng = np.random.default_rng()

        self.teams = [[f"snake_{team_id}_{snake_id}" for snake_id in range(self.team_snakes_count)]
                      for team_id in range(self.teams_count)]
        self.possible_agents = sum(self.teams, [])

        self.action_spaces = {agent: Action for agent in self.possible_agents}

        self.renderer = None

        self.metadata = {"render.mode": ["human"]}

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.rng = np.random.default_rng()
        self.renderer = None

        self.agents = self.possible_agents.copy()

        Snake._color = FOOD_COLOR
        self.snakes = {}

        for idx, team in enumerate(self.teams):
            for agent in team:
                self.snakes[agent] = Snake(self.random_empty_position(), team=idx)

        self.food = {self.random_empty_position() for _ in range(self.min_food_count)}
        self.food_spawn_timer = 0

        self.reward = {agent: 0.0 for agent in self.agents}
        self.eliminated_agents = set()

        observation = Observation()
        observation.field = self.as_array()
        observation = {agent: observation for agent in self.agents}
        info = {agent: {} for agent in self.agents}

        return observation, info, {}

    def step(self, action: dict[str, Action]):

        self.reward = {agent: 0.0 for agent in self.agents}

        for agent in self.agents:
            self.snakes[agent].step(action[agent])

        self.check_health()
        self.check_border_collisions()
        self.check_food_collisions()
        self.check_self_collisions()
        self.check_head_head_collisions()
        self.check_another_collisions()

        self.spawn_food()

        end_game = self.check_end_game()

        observation = Observation()
        observation.field = self.as_array()
        observation = {agent: observation for agent in action}
        if end_game:
            terminated = {agent: True for agent in action}
        else:
            terminated = {agent: agent in self.eliminated_agents for agent in action}
        truncated = {agent: False for agent in action}
        info = {agent: {} for agent in action}

        return observation, self.reward, terminated, truncated, info, {}

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
                    self.reward[other_agent] = 1.0
        self.eliminate(eliminate_agents)

    def eliminate(self, eliminate_agent):
        for agent in eliminate_agent:
            self.reward[agent] = -1.0
            self.agents.remove(agent)
            self.snakes.pop(agent)
            self.eliminated_agents.add(agent)

    def as_array(self):

        field = np.zeros(self.size)
        for agent, snake in self.snakes.items():
            for x, y in snake.body:
                field[x][y] = snake.color

        for x, y in self.food:
            field[x][y] = FOOD_COLOR

        return field

    def random_empty_position(self):
        rng = np.random.default_rng()
        field = self.as_array()
        empty_indices = np.argwhere(field == EMPTY_COLOR)
        index = rng.choice(empty_indices, axis=0)
        return tuple(index)

    def observation_space(self, agent: AgentID):
        return Observation

    def action_space(self, agent: AgentID):
        return Action

    def render(self) -> None:

        if self.renderer is None:
            self.renderer = BattleSnakeRenderer(self)

        self.renderer.render()

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
