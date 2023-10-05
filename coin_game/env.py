from __future__ import annotations

import random
from copy import deepcopy

import gymnasium
import numpy as np
import pettingzoo
from pettingzoo.utils.env import AgentID, ObsType, ActionType


class CoinGame(pettingzoo.ParallelEnv):

    def __init__(self, agents_num, prev_actions_num):
        self.agents_num = agents_num
        self.prev_actions_num = prev_actions_num
        self.possible_agents = [f"agent_{i}" for i in range(agents_num)]
        self.metadata = {"render.mode": ["ascii"]}
        self.prev_actions = None

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:
        self.agents = deepcopy(self.possible_agents)
        self.prev_actions = {agent: [random.randint(0, 1) for _ in range(self.prev_actions_num)] for agent in self.agents}
        return self._observation(), {agent: {} for agent in self.agents}

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.MultiBinary(1)

    def observation_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.MultiBinary((self.agents_num, self.prev_actions_num, ))

    def _observation(self):
        observation = {}
        for agent in self.agents:
            observation[agent] = np.array(list(self.prev_actions.values()))
        return observation

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        values = actions.values()
        if all(values):
            reward = {agent: 0 for agent in self.agents}
        elif any(values):
            reward = {agent: 3 if actions[agent] else -1 for agent in self.agents}
        else:
            reward = {agent: 2 for agent in self.agents}

        for agent, act in actions.items():
            self.prev_actions[agent].pop(0)
            self.prev_actions[agent].append(act[0])

        return (self._observation(),
                reward,
                {agent: False for agent in self.agents},
                {agent: False for agent in self.agents},
                {agent: {} for agent in self.agents})

    def render(self) -> None | np.ndarray | str | list:

        ACTIONS = ["coop", "cheat", "none"]

        for agent in self.agents:
            print(f"Agent {agent}: {' '.join(map(lambda action: ACTIONS[action], self.prev_actions[agent]))}")
