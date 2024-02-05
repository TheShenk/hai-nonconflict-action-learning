from __future__ import annotations

import bz2
import argparse
import pathlib
import sqlite3
from typing import Any, SupportsFloat

import cloudpickle
import gymnasium
import minari
import numpy as np
from gymnasium.core import ObsType, ActType

parser = argparse.ArgumentParser()
parser.add_argument("path", type=pathlib.Path)
parser.add_argument("save_path", type=pathlib.Path)
parser.add_argument("--user", type=str)
parser.add_argument("--task", type=str)
parser.add_argument("--agent", type=str)
args = parser.parse_args()

path = args.path


class RecordEnv(gymnasium.Env):

    def __init__(self, data, agent_id):
        self.data = data
        self.agent_id = agent_id
        self.current_step = iter(data)

        observation = np.array(data[0]["prev_obs"][agent_id])
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=observation.shape, dtype=observation.dtype)

        action = np.array(data[0]['action'][agent_id])
        self.action_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=action.shape, dtype=action.dtype)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return self.data[0]["prev_obs"][self.agent_id], self.data[0]["info"][self.agent_id]

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        current_step_data = next(self.current_step)
        return (current_step_data["prev_obs"][self.agent_id],
                current_step_data["reward"][self.agent_id],
                current_step_data["done"][self.agent_id],
                False, current_step_data["info"][self.agent_id])


with sqlite3.connect(path / "dataset.sqlite") as con:
    cur = con.cursor()
    res = cur.execute("select episode_id from users "
                      "inner join episodes on users.environment_id = episodes.environment_id "
                      "where permanent_user_id = ? and task_id = ? and agent_id = ?",
                      (args.user, args.task, args.agent))
    data = res.fetchall()

episode_data = []
for episode_id, in data:
    episode = bz2.open(path.parent / f"{episode_id}.pickle.bz2")
    episode = cloudpickle.load(episode)
    episode_data.extend(episode)

env = RecordEnv(episode_data, args.agent)
env.spec = gymnasium.envs.registration.EnvSpec(id=args.task)
env = minari.DataCollectorV0(env=env)

env.reset()
for step in episode_data: env.step(step['action'][args.agent])

env.save_to_disk(args.save_path, {
    "dataset_id": f"{args.task}-human-v0",
    "minari_version": "~=0.4.1"
})
env.close()
