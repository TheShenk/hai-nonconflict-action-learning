from __future__ import annotations

import gymnasium
import numpy as np
import pettingzoo
import shimmy
import torch as th
from ray.rllib import MultiAgentEnv


class RolloutInfo:

    def __init__(self, policy, prev_action):
        self.policy = policy
        self.prev_action = prev_action
        self.state = th.from_numpy(np.zeros((2,)))  # TODO: what state size depends on?

    def predict(self, obs):
        action, state, info = self.policy.compute_single_action(obs=obs,
                                                                state=self.state,
                                                                prev_action=self.prev_action)
        self.prev_action = action
        self.state = state
        return action


class SingleAgent(gymnasium.Env):

    def step(self, action):
        total_action = {agent_id: self.presetted_policies[agent_id].predict(self.observation[agent_id])
                        for agent_id in self.env.agents if agent_id in self.presetted_policies}

        total_action[self.controlled_agent_id] = action
        self.observation, rewards, dones, infos = self.env.step(total_action)
        return self.observation[self.controlled_agent_id]['obs'], rewards[self.controlled_agent_id], dones[
            self.controlled_agent_id], False, infos[self.controlled_agent_id]

    def reset(self, seed=None, options=None):
        self.observation = self.env.reset()
        return self.observation[self.controlled_agent_id]['obs'], {}

    def render(self, mode="human"):
        self.env.render(mode)

    def __init__(self, env: MultiAgentEnv, presetted_policies: dict, controlled_agent_id):
        super().__init__()
        self.env = env
        self.observation = None
        self.presetted_policies = {
            agent_id: RolloutInfo(presetted_policies[agent_id], np.zeros(self.env.action_space.shape))
            for agent_id in presetted_policies}
        self.controlled_agent_id = controlled_agent_id
        self.observation_space = shimmy.openai_gym_compatibility._convert_space(self.env.observation_space['obs'])
        self.action_space = shimmy.openai_gym_compatibility._convert_space(self.env.action_space)

    def close(self):
        self.env.close()


def exclude(dictionary: dict, keys):
    excluded_dict = {}
    for key, item in dictionary.items():
        if key not in keys:
            excluded_dict[key] = item
    return excluded_dict


def exclude_list(elements: list, keys):
    return [el for el in elements if el not in keys]


class PresetAgents(pettingzoo.ParallelEnv):

    def __init__(self, env: pettingzoo.ParallelEnv, preset_policies: dict):
        super().__init__()
        self.env = env
        self.observation = None
        self.preset_policies = preset_policies
        self.observation_spaces = exclude(self.env.observation_spaces, preset_policies.keys())
        self.action_spaces = exclude(self.env.action_spaces, preset_policies.keys())
        self.metadata = self.env.metadata
        self.possible_agents = exclude_list(self.env.possible_agents, self.preset_policies.keys())
        self.agents = self.possible_agents

    def step(self, action):
        total_action = {agent_id: self.preset_policies[agent_id].predict(self.observation[agent_id])
                        for agent_id in self.preset_policies} | action

        self.observation, rewards, terminals, infos = self.env.step(total_action)
        return exclude(self.observation, self.preset_policies.keys()), \
            exclude(rewards, self.preset_policies.keys()), \
            exclude(terminals, self.preset_policies.keys()), \
            exclude(infos, self.preset_policies)

    def reset(self):
        self.observation = self.env.reset()
        return exclude(self.observation, self.preset_policies.keys())

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()
