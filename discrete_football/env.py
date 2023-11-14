from __future__ import annotations

import pettingzoo
from pettingzoo.utils.env import AgentID, ObsType, ActionType

SHIFT_BY_ACTION = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0), (0, 0)]


class FieldObject:

    def __init__(self, init_position = (0, 0)):
        self.position = init_position
        self.prev_shift = (0, 0)

    def with_shift(self, shift):
        return (self.position[i] + shift[i] for i in range(len(shift)))

    def shift(self, shift):
        self.position = self.with_shift(shift)
        self.prev_shift = shift


class DiscreteFootball(pettingzoo.ParallelEnv):

    def __init__(self, players_n=2, size=(33, 17), render_mode="human"):

        self.ball = None
        self.players = None

        self.teams_count = 2
        self.teams = [[f"player_{team_id}_{snake_id}" for snake_id in range(self.team_snakes_count)]
                      for team_id in range(self.teams_count)]
        self.possible_agents = sum(self.teams, [])
        self.render_mode = render_mode

        self.metadata = {"render.mode": ["human"]}

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:

        self.agents = self.possible_agents.copy()
        self.players = {agent_id: FieldObject() for agent_id in self.agents}
        self.ball = FieldObject()

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:

        field_objects = {player.position for player in self.players.values()}
        ball_position = self.ball.position
        for agent, act in actions.items():
            player = self.players[agent]
            shift = SHIFT_BY_ACTION[act]
            if player.with_shift(shift) not in field_objects:
                if player.position == ball_position:
                    if self.ball.with_shift(shift) not in field_objects:
                        self.ball.shift(shift)
                        player.shift(shift)
                else:
                    player.shift(shift)

        field_objects = {player.position for player in self.players.values()}
        observation = sum(map(lambda p: list(p.position), field_objects), [])



