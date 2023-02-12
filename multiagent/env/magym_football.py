import gym
from ma_gym.envs.utils.action_space import MultiAgentActionSpace

from gym_futbol.envs_v1 import Futbol
import ma_gym

from multiagent.action_combiners import NON_VEC_DISCRETE_COMBINER, NON_VEC_COMBINER


class MaGymFootball(gym.Env):

    def __init__(self, env: Futbol):
        self.env = env
        self.n_agents = 2
        self.action_space = MultiAgentActionSpace([gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))
                                                   for _ in range(self.n_agents)])
        self.observation_space = MultiAgentActionSpace([gym.spaces.Box(low=-1.0, high=1.0, shape=(20,))
                                                   for _ in range(self.n_agents)])

    def get_agent_obs(self):
        return [self.env.observation for _ in range(self.n_agents)]

    def reset(self):
        self.env.reset()
        return self.get_agent_obs()

    def step(self, action_n):
        # assert len(action_n) == self.n_agents

        combiner_function = NON_VEC_DISCRETE_COMBINER if self.env.action_space_type[
                                                             0] == "discrete" else NON_VEC_COMBINER
        combined_act = action_n

        obs, rew, done, info = self.env.step(combined_act)

        observation_n = [obs for _ in range(self.n_agents)]
        reward_n = [rew for _ in range(self.n_agents)]
        done_n = [done for _ in range(self.n_agents)]

        return observation_n, reward_n, done_n, info

    def render(self, mode="human"):
        pass