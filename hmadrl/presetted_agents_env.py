import gym
import numpy as np
from ray.rllib import MultiAgentEnv


class RolloutInfo:

    def __init__(self, policy, prev_action):
        self.policy = policy
        self.prev_action = prev_action
        self.state = np.zeros((2,))  # TODO: what state size depends on?

    def predict(self, obs):
        action, state, info = self.policy.compute_single_action(obs=obs,
                                                                state=self.state,
                                                                prev_action=self.prev_action)
        self.prev_action = action
        self.state = state
        return action


class PreSettedAgentsEnv(gym.Env):

    def step(self, action):
        total_action = {agent_id: self.presetted_policies[agent_id].predict(self.observation[agent_id])
                        for agent_id in self.presetted_policies}

        total_action[self.controlled_agent_id] = action
        self.observation, rewards, terminals, infos = self.env.step(total_action)
        return self.observation[self.controlled_agent_id]['obs'], rewards[self.controlled_agent_id], terminals[
            self.controlled_agent_id], infos[self.controlled_agent_id]

    def reset(self):
        self.observation = self.env.reset()
        return self.observation[self.controlled_agent_id]['obs']

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
        self.observation_space = self.env.observation_space['obs']
        self.action_space = self.env.action_space

    def close(self):
        self.env.close()


class PreSettedAgentsMultiEnv(MultiAgentEnv):

    def _only_real(self, object: dict):
        return {agent_id: sub_object for agent_id, sub_object in object.items() if agent_id not in self.presetted_policies}

    def step(self, action):
        presetted_action = {agent_id: self.presetted_policies[agent_id].predict(self.observation[agent_id]['obs'])[0]
                            for agent_id in self.presetted_policies}

        total_action = action | presetted_action
        self.observation, rewards, terminals, infos = self.env.step(total_action)
        return self._only_real(self.observation), self._only_real(rewards), self._only_real(terminals), self._only_real(infos)

    def reset(self):
        self.observation = self.env.reset()
        return self._only_real(self.observation)

    def render(self, mode="human"):
        self.env.render(mode)

    def __init__(self, env: MultiAgentEnv, presetted_policies: dict):
        super().__init__()
        self.env = env
        self.observation = None
        self.presetted_policies = {
            agent_id: presetted_policies[agent_id]
            for agent_id in presetted_policies}
        self.agents = self.env.agents
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def close(self):
        self.env.close()

    def get_env_info(self):
        env_info = self.env.get_env_info()
        env_info['num_agents'] = env_info['num_agents'] - len(self.presetted_policies)
        return env_info