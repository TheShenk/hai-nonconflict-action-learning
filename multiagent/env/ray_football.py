from typing import Tuple

import gym
from gym_futbol.envs_v1 import Futbol

import numpy as np

from ray.rllib.env import PettingZooEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune import register_env
from ray.rllib.utils.typing import MultiAgentDict

from agents.simple_attacking_agent import SimpleAttackingAgent
from agents.simple_goalkeeper_agent import SimpleGoalkeeperAgent

from multiagent.env.multiagent_football import MultiAgentFootball
from multiagent.multi_model_agent import MultiModelAgent
from multiagent.action_combiners import NON_VEC_COMBINER, NON_VEC_DISCRETE_COMBINER


class RayFootballProxy(gym.Env):

    def render(self, mode="human"):
        pass

    def __init__(self, env: Futbol):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(8,))

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info =  self.env.step(np.reshape(action, (2,4)))
        obs = np.clip(obs, -1.0, 1.0)
        return obs, rew, done, info


class RayMultiAgentFootball(MultiAgentEnv):

    def __init__(self, env: Futbol):
        super().__init__()
        self.env = env
        self.agents = [f"player_{r}" for r in range(env.number_of_player)]

        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Dict({agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in self.agents})

        self._agent_ids = [f"player_{r}" for r in range(env.number_of_player)]
        # self._action_space_in_preferred_format = {agent: gym.spaces.Box(low=-1.0, high=1.0, shape=(4,)) for agent in self,agents}

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        observations = {agent: obs for agent in self.agents}
        return observations

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        total_act = [action_dict[agent] for agent in self.agents]
        combiner_function = NON_VEC_DISCRETE_COMBINER if self.env.action_space_type[
                                                             0] == "discrete" else NON_VEC_COMBINER
        combined_act = combiner_function(total_act)

        obs, rew, done, info = self.env.step(combined_act)
        obs = np.clip(obs, -1.0, 1.0)

        observations = {agent: obs for agent in self.agents}
        rewards = {agent: rew for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones.update({"__all__": done})
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos



def create_football_hca(env_config):
    env = Futbol(number_of_player=2, action_space_type=["box", "box"])
    env.set_team_b_model(MultiModelAgent(env, static_models=[
        SimpleAttackingAgent(env, 0),
        SimpleGoalkeeperAgent(env, 1)
    ]))
    return RayFootballProxy(env)

def create_ma_football_hca(env_config):
    env = Futbol(number_of_player=2, action_space_type=["box", "box"])
    env.set_team_b_model(MultiModelAgent(env, static_models=[
        SimpleAttackingAgent(env, 0),
        SimpleGoalkeeperAgent(env, 1)
    ]))
    return RayMultiAgentFootball(env)

register_env("football-hca", create_football_hca)
register_env("ma-football-hca", create_ma_football_hca)