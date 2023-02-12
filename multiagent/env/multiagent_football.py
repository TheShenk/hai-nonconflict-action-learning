from typing import Optional, Tuple, Dict, Any

import gym.spaces
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from pettingzoo.utils.env import ObsType, ActionType

from gym_futbol.envs_v1 import Futbol
from multiagent.action_combiners import NON_VEC_COMBINER, NON_VEC_DISCRETE_COMBINER


class MultiAgentFootball(AECEnv):

    def __init__(self, env: Futbol):
        super().__init__()
        self.num_moves = 0
        self.env = env

        self.possible_agents = [r for r in range(env.number_of_player)]
        self.agents = self.possible_agents
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.action_spaces = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in self.possible_agents}
        self.observation_spaces = {
            agent: env.observation_space for agent in self.possible_agents
        }

    def step(self, action: ActionType):

        if self.dones[self.agent_selection] or self.truncations[self.agent_selection]:
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        self.current_action[agent] = action

        if self._agent_selector.is_last():

            total_act = [self.current_action[agent] for agent in self.possible_agents]
            combiner_function = NON_VEC_DISCRETE_COMBINER if self.env.action_space_type[
                                                                 0] == "discrete" else NON_VEC_COMBINER
            combined_act = combiner_function(total_act)

            obs, rew, done, info = self.env.step(combined_act)

            self.rewards = {agent: rew for agent in self.possible_agents}
            self.observations = {agent: obs for agent in self.possible_agents}
            self.dones = {agent: done for agent in self.possible_agents}
            self.terminations = self.dones
            self.truncations = {agent: False for agent in self.possible_agents}
            self.infos = {agent: {} for agent in self.possible_agents}

        else:
            # no rewards are allocated until both players give an action
            self._clear_rewards()

        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

    def reset(self, seed: Optional[int] = None, return_info: bool = False, options: Optional[dict] = None):
        obs = self.env.reset()
        observations = {agent: obs for agent in self.possible_agents}

        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.terminations = self.dones
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: [] for agent in self.agents}
        self.current_action = {agent: [] for agent in self.possible_agents}
        self.num_moves = 0

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()

        if not return_info:
            return observations
        else:
            infos = {agent: {} for agent in self.possible_agents}
            return observations, infos

    def seed(self, seed: Optional[int] = None) -> None:
        pass

    def observe(self, agent: str) -> Optional[ObsType]:
        return self.env.observation

    def render(self) -> None | np.ndarray | str | list:
        pass

    def state(self) -> np.ndarray:
        return self.env.observation

    def last(
        self, observe: bool = True
    ) -> Tuple[Optional[ObsType], float, bool, Dict[str, Any]]:
        obs, rew, done, trunc, info = super().last(observe)
        return obs, rew, done, info