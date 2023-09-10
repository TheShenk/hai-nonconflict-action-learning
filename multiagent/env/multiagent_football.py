from __future__ import annotations

from typing import Optional, Tuple, Dict, Any

import gym.spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pettingzoo import AECEnv, ParallelEnv
from pettingzoo.utils import agent_selector

from multiagent.action_combiners import NON_VEC_COMBINER, NON_VEC_DISCRETE_COMBINER
from multiagent.env.football import TwoSideFootball


class MultiAgentFootball(ParallelEnv):

    def __init__(self, env: TwoSideFootball, render_mode="human"):
        self.env = env
        self.red_agents = [f"red_{i}" for i in range(env.number_of_player)]
        self.blue_agents = [f"blue_{i}" for i in range(env.number_of_player)]
        self.possible_agents = self.red_agents + self.blue_agents
        self.agents = self.possible_agents
        self.render_mode = render_mode

        self.surface = None

        self.observation_spaces = {agent: self.env.observation_space for agent in self.agents}

        agent_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) if self.env.action_space_type[0] == 'box' \
            else gym.spaces.Discrete(25)
        self.action_spaces = {agent: agent_action_space for agent in self.agents}

        self.combiner_function = NON_VEC_COMBINER if self.env.action_space_type[0] == "box" \
            else NON_VEC_DISCRETE_COMBINER

        self.metadata = {"render.modes": ["human", "rgb_array"]}

    def reset(self):
        obs = self.env.reset()
        inverse_obs = self.env.inverse_obs
        observations = {agent: obs for agent in self.red_agents} | \
                       {agent: inverse_obs for agent in self.blue_agents}
        return observations, {}

    def step(self, actions):
        red_action = [actions[agent] for agent in self.red_agents]
        red_action = np.clip(red_action, -1.0, 1.0)
        self.env.act(red_action)

        blue_action = [actions[agent] for agent in self.blue_agents]
        blue_action = np.clip(blue_action, -1.0, 1.0)
        self.env.inverted_act(blue_action)

        obs, inverse_obs, done, info = self.env.commit()
        obs = np.clip(obs, -1.0, 1.0)
        inverse_obs = np.clip(inverse_obs, -1.0, 1.0)

        red_reward = self.env.calculate_left_reward()
        blue_reward = self.env.calculate_right_reward()

        observations = {agent: obs.copy() for agent in self.red_agents} | \
                       {agent: inverse_obs.copy() for agent in self.blue_agents}
        rewards = {agent: red_reward / 1000 for agent in self.red_agents} | \
                  {agent: blue_reward / 1000 for agent in self.blue_agents}
        terminated = {agent: done for agent in self.agents}
        terminated.update({"__all__": done})
        truncated = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, terminated, truncated, infos

    def render(self):
        if not self.surface:
            RES = WIDTH, HEIGHT = 600, 400
            FPS = 24

            if self.render_mode == "human":
                pygame.init()
                pygame.key.set_repeat(1, 1)
                self.clock = pygame.time.Clock()

            self.surface = pygame.display.set_mode(RES)

            translation = (4, 2)
            scale_factor = min(WIDTH / (self.env.width + translation[0] * 2),
                               HEIGHT / (self.env.height + translation[1] * 2))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)
            self.draw_options.transform = pymunk.Transform.scaling(scale_factor) @ pymunk.Transform.translation(
                translation[0], translation[1])
            self.fps = FPS

        self.surface.fill("black")
        self.env.space.debug_draw(self.draw_options)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.fps)
        else:
            return pygame.surfarray.array3d(self.surface)

    def close(self):
        pygame.quit()


class AECMultiAgentFootball(AECEnv):

    def __init__(self, env):
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

    def step(self, action):

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

    def observe(self, agent: str):
        return self.env.observation

    def render(self) -> None | np.ndarray | str | list:
        pass

    def state(self) -> np.ndarray:
        return self.env.observation

    def last(
        self, observe: bool = True
    ) -> Tuple[Any, float, bool, Dict[str, Any]]:
        obs, rew, done, trunc, info = super().last(observe)
        return obs, rew, done, info

    def close(self):
        self.env.close()
