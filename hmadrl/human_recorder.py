import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class HumanRecorder(MultiAgentEnv):

    def __init__(self, env: MultiAgentEnv, human_agent_id, trajectory_file, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.human_agent_id = human_agent_id
        self.trajectory_file = trajectory_file

        self.human_observations = []
        self.human_actions = []
        self.human_rewards = []
        self.human_infos = []
        self.terminals = []

        self.episode_observations = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_infos = []
        self.episode_terminal = False

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agents = self.env.agents

    def step(self, actions):
        observations, reward, terminated, info = self.env.step(actions)

        self.episode_actions.append(actions[self.human_agent_id])
        self.episode_observations.append(observations[self.human_agent_id]['obs'])
        self.episode_rewards.append(reward[self.human_agent_id])
        self.episode_infos.append(info[self.human_agent_id])
        self.episode_terminal = terminated[self.human_agent_id]
        return observations, reward, terminated, info

    def reset(self):
        observation = self.env.reset()
        self.human_actions.append(self.episode_actions)
        self.human_observations.append(self.episode_observations)
        self.human_rewards.append(self.episode_rewards)
        self.human_infos.append(self.episode_infos)
        self.terminals.append(self.episode_terminal)
        return observation

    def close(self):
        np.savez(self.trajectory_file,
                 actions=np.array(self.human_actions),
                 observations=np.array(self.human_observations),
                 rewards=np.array(self.human_rewards),
                 infos=np.array(self.human_infos),
                 terminal=np.array(self.terminals))
        print(np.array(self.human_actions))
        self.env.close()

    def get_env_info(self):
        return self.env.get_env_info()

    def render(self, mode=None) -> None:
        return self.env.render(mode)
