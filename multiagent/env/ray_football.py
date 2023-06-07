from typing import Tuple

import gym
import pygame
import pymunk

from gym_futbol.envs_v1.futbol_env import Futbol, TIME_STEP

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

import pymunk.pygame_util

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
        self.surface = None
        self.agents = [f"player_{r}" for r in range(env.number_of_player)]

        self.observation_space = gym.spaces.Dict({"obs": self.env.observation_space})
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,))

        self._agent_ids = [f"player_{r}" for r in range(env.number_of_player)]

    def reset(self) -> MultiAgentDict:
        obs = self.env.reset()
        observations = {agent: {"obs": obs} for agent in self.agents}
        return observations

    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:

        total_act = [action_dict[agent] for agent in self.agents]
        combiner_function = NON_VEC_DISCRETE_COMBINER if self.env.action_space_type[0] == "discrete" \
            else NON_VEC_COMBINER
        combined_act = combiner_function(total_act)


        obs, rew, done, info = self.env.step(combined_act)
        obs = np.clip(obs, -1.0, 1.0)

        observations = {agent: {"obs": obs.copy()} for agent in self.agents}
        rewards = {agent: rew/1000 for agent in self.agents}
        dones = {agent: done for agent in self.agents}
        dones.update({"__all__": done})
        infos = {agent: {} for agent in self.agents}

        return observations, rewards, dones, infos

    def get_env_info(self):
        env_info = {
            "space_obs": self.observation_space,
            "space_act": self.action_space,
            "num_agents": self.env.number_of_player,
            "episode_limit": int(self.env.total_time/TIME_STEP),
            "policy_mapping_info": {"all_scenario":
                                        {"description": "one team smart",
                                         "team_prefix": ("player_",),
                                         "all_agents_one_policy": True,
                                         "one_agent_one_policy": True}
                                    }
        }
        return env_info

    def render(self, mode=None):
        if not pygame.get_init():
            RES = WIDTH, HEIGHT = 600, 400
            FPS = 24

            pygame.init()
            pygame.key.set_repeat(1, 1)
            self.surface = pygame.display.set_mode(RES)
            self.clock = pygame.time.Clock()

            translation = (4, 2)
            scale_factor = min(WIDTH / (self.env.width + translation[0] * 2), HEIGHT / (self.env.height + translation[1] * 2))
            self.draw_options = pymunk.pygame_util.DrawOptions(self.surface)
            self.draw_options.transform = pymunk.Transform.scaling(scale_factor) @ pymunk.Transform.translation(
                translation[0], translation[1])
            self.fps = FPS

        self.surface.fill("black")
        self.env.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def close(self):
        if pygame.get_init():
            pygame.quit()

def create_football_hca(env_config):
    env = Futbol(number_of_player=2, action_space_type=["box", "box"])
    env.set_team_b_model(MultiModelAgent(env, static_models=[
        SimpleAttackingAgent(env, 0),
        SimpleGoalkeeperAgent(env, 1)
    ]))
    return RayFootballProxy(env)

def create_ma_football_hca(env_config):
    print("create_ma_football_hca", env_config)
    env = Futbol(number_of_player=2, action_space_type=["box", "box"])
    env.set_team_b_model(MultiModelAgent(env, static_models=[
        SimpleAttackingAgent(env, 0),
        SimpleGoalkeeperAgent(env, 1)
    ]))
    return RayMultiAgentFootball(env)

register_env("football-hca", create_football_hca)
register_env("ma-football-hca", create_ma_football_hca)