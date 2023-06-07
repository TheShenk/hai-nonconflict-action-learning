import gym
from ray.rllib import MultiAgentEnv


class PreSettedAgentsEnv(gym.Env):

    def step(self, action):
        total_action = {agent_id: self.presetted_policies[agent_id].compute_single_action(self.observation[agent_id])[0]
                        for agent_id in self.presetted_policies}
        total_action[self.controlled_agent_id] = action
        self.observation, rewards, terminals, infos = self.env.step(total_action)
        return self.observation[self.controlled_agent_id]['obs'], rewards[self.controlled_agent_id], terminals[self.controlled_agent_id], infos[self.controlled_agent_id]

    def reset(self):
        self.observation = self.env.reset()
        return self.observation[self.controlled_agent_id]['obs']

    def render(self, mode="human"):
        self.env.render(mode)

    def __init__(self, env: MultiAgentEnv, presetted_policies: dict, controlled_agent_id):
        super().__init__()
        self.env = env
        self.observation = None
        self.presetted_policies = presetted_policies
        self.controlled_agent_id = controlled_agent_id
        self.observation_space = self.env.observation_space['obs']
        self.action_space = self.env.action_space

    def close(self):
        self.env.close()
