import json
import pathlib
import random
from typing import Dict

import numpy as np

import torch
from cloudpickle import cloudpickle
from ray.rllib.policy.policy import PolicySpec


class MARLlibPretrained:

    def __init__(self, checkpoint_path, policy_id):

        with open(checkpoint_path, 'rb') as checkpoint_file:
            checkpoint = cloudpickle.load(checkpoint_file)

        params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'
        with open(params_path, 'r') as params_file:
            params = json.load(params_file)

        random.seed(params["seed"])
        np.random.seed(params["seed"])
        torch.manual_seed(params["seed"])

        worker = cloudpickle.loads(checkpoint['worker'])
        policies: Dict[str, PolicySpec] = worker['policy_specs']
        observation_space = policies[policy_id].observation_space
        action_space = policies[policy_id].action_space

        params["model"]["custom_model"] = "current_model"
        params["multiagent"]["policies"] = policies
        params["model"]["custom_model_config"]["space_obs"] = observation_space.original_space
        params["model"]["custom_model_config"]["space_act"] = action_space

        policy_state = worker["state"][policy_id]
        self.policy = policies[policy_id].policy_class(obs_space=observation_space, action_space=action_space, config=params)
        self.policy.set_state(policy_state)

    def compute_action(self, obs):
        return self.policy.compute_single_action(obs)[0]