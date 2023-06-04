import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class HumanRecorder(MultiAgentEnv):

    def __init__(self, env: MultiAgentEnv, human_agent_id, trajectory_file, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.human_agent_id = human_agent_id
        self.trajectory_file = trajectory_file
        self.human_actions = np.array([])
        self.human_observations = np.array([])

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agents = self.env.agents

    def step(self, actions):
        self.human_actions = np.append(self.human_actions, actions[self.human_agent_id])
        observations, reward, terminated, info = self.env.step(actions)
        self.human_observations = np.append(self.human_observations, observations[self.human_agent_id]['obs'])
        return observations, reward, terminated, info

    def reset(self):
        observation = self.env.reset()
        np.savez(self.trajectory_file, actions=self.human_actions, observations=self.human_observations)
        return observation

    def close(self):
        self.env.close()
        np.savez(self.trajectory_file, actions=self.human_actions, observations=self.human_observations)

    def get_env_info(self):
        return self.env.get_env_info()

    def render(self, mode=None) -> None:
        return self.env.render(mode)
