import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class HumanRecorder(MultiAgentEnv):

    def __init__(self, env: MultiAgentEnv, human_agent_id, trajectory_file, **kwargs):
        print('init')
        super().__init__(**kwargs)
        self.env = env
        self.human_agent_id = human_agent_id
        self.trajectory_file = trajectory_file

        self.actions_record = []
        self.observations_record = []
        self.rewards_record = []
        self.dones_record = []
        self.infos_record = []

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.agents = self.env.agents

    def step(self, actions):
        observation, reward, done, info = self.env.step(actions)

        self.actions_record.append(actions[self.human_agent_id])
        self.observations_record.append(observation[self.human_agent_id]['obs'])
        self.rewards_record.append(reward[self.human_agent_id])
        self.dones_record.append(done[self.human_agent_id])
        self.infos_record.append(info[self.human_agent_id])

        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.observations_record.append(observation[self.human_agent_id]['obs'])
        return observation

    def close(self):
        print('close:', len(self.actions_record))
        np.savez(self.trajectory_file,
                 actions=np.array(self.actions_record),
                 observations=np.array(self.observations_record),
                 rewards=np.array(self.rewards_record),
                 infos=np.array(self.infos_record),
                 dones=np.array(self.dones_record))
        self.env.close()

    def get_env_info(self):
        return self.env.get_env_info()

    def render(self, mode=None) -> None:
        return self.env.render(mode)
