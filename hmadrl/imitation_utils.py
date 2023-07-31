import re

import numpy as np
import stable_baselines3.common.base_class
from imitation.data.types import TrajectoryWithRew
from ray.rllib.utils.torch_ops import convert_to_torch_tensor
from stable_baselines3.common.policies import BasePolicy


def make_trajectories(trajectories_data: np.ndarray):
    actions = trajectories_data['actions']
    observations = trajectories_data['observations']
    rewards = trajectories_data['rewards']
    dones = trajectories_data['dones']

    trajectories = []
    done_indexes, = np.where(dones)
    observation_shift = 0

    for previous_done, current_done in zip(np.append([0], done_indexes[:-1]), done_indexes):

        trajectories.append(TrajectoryWithRew(acts=actions[previous_done:current_done],
                                              obs=observations[previous_done + observation_shift:current_done + observation_shift + 1],
                                              rews=rewards[previous_done:current_done],
                                              infos=np.empty((current_done-previous_done,)),
                                              terminal=True))
        observation_shift += 1

    return trajectories


def init_as_multiagent(imitation_policy: BasePolicy, rllib_policy):

    rllib_weights = rllib_policy.get_weights()
    imitation_weights = imitation_policy.state_dict()

    converter = {
        r"p_encoder\.encoder\.(\d+)\._model\.0.(\w+)": "mlp_extractor.policy_net",
        r"vf_encoder\.encoder\.(\d+)\._model\.0.(\w+)": "mlp_extractor.value_net",
        # r"p_branch._model.0.(\w+)": "action_net",
        r"vf_branch._model.0.(\w+)": "value_net"
    }

    for layer_re, imitation_layer_template in converter.items():
        layer_re = re.compile(layer_re)
        for rllib_layer in rllib_weights.keys():
            matching = layer_re.match(rllib_layer)
            if matching:
                if len(matching.groups()) == 2:
                    layer_number, layer_type = matching.groups()
                    imitation_layer = f"{imitation_layer_template}.{int(layer_number) * 2}.{layer_type}"
                else:
                    layer_type = matching.groups()[0]
                    imitation_layer = f"{imitation_layer_template}.{layer_type}"
                imitation_weights[imitation_layer] = convert_to_torch_tensor(rllib_weights[rllib_layer])

    imitation_policy.load_state_dict(imitation_weights)
