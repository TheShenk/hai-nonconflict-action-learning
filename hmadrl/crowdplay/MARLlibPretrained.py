import json
import pathlib
import random
from typing import Dict

import numpy as np

import torch
from cloudpickle import cloudpickle
from ray.rllib.agents import with_common_config
from ray.rllib.policy.policy import PolicySpec


class MARLlibPretrained:

    def __init__(self, checkpoint_path, policy_id, first_action = None):

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

        config = with_common_config(params)
        config["model"]["custom_model"] = "current_model"
        config["multiagent"]["policies"] = policies
        config["model"]["custom_model_config"]["space_obs"] = observation_space.original_space
        config["model"]["custom_model_config"]["space_act"] = action_space

        policy_state = worker["state"][policy_id]
        self.policy = policies[policy_id].policy_class(obs_space=observation_space, action_space=action_space, config=config)
        self.policy.set_state(policy_state)
        self.first_action = first_action

    def compute_action(self, obs):
        if self.first_action is not None:
            action, self.first_action = self.first_action, None
            return action
        return self.policy.compute_single_action(obs)[0]
