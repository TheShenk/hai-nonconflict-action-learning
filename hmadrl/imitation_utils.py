import re
from typing import Dict, Type

import numpy as np
from imitation.data.types import TrajectoryWithRew
from imitation.rewards.reward_nets import AddSTDRewardWrapper, BasicRewardNet, CnnRewardNet, BasicShapedRewardNet, \
    RewardEnsemble, RewardNet
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


REWARD_NET_WRAPPER_REGISTRY = {
    # "NormalizedRewardNet": NormalizedRewardNet, #TODO: how create layer from settings?
    # "ShapedRewardNet": ShapedRewardNet, # TODO: how create callable from settings?
    "AddSTDRewardWrapper": AddSTDRewardWrapper
}


REWARD_NET_REGISTRY: Dict[str, Type[RewardNet]] = {
    "BasicRewardNet": BasicRewardNet,
    "CnnRewardNet": CnnRewardNet,
    "BasicShapedRewardNet": BasicShapedRewardNet,
    "RewardEnsemble": RewardEnsemble
}


def register_reward_net_wrapper(key, wrapper_creator_fn):
    REWARD_NET_WRAPPER_REGISTRY[key] = wrapper_creator_fn


def register_reward_net(key, net_creator_fn):
    REWARD_NET_REGISTRY[key] = net_creator_fn


def get_cls_name(obj):
    if isinstance(obj, dict):
        return list(obj.keys())[0]
    else:
        return obj


def get_cls_kwargs(obj, cls_name):
    if isinstance(obj, dict):
        return obj[cls_name]
    else:
        return {}


def create_reward_net(reward_net_settings, observation_space, action_space):

    reward_net_cls_name = get_cls_name(reward_net_settings) if reward_net_settings else "BasicRewardNet"
    reward_net_kwargs = get_cls_kwargs(reward_net_settings, reward_net_cls_name)

    if reward_net_cls_name == "RewardEnsemble":
        assert "members" in reward_net_kwargs, "Ensemble must be initiated with list of nets (member argument)"
        members = [create_wrapped_reward_net(member_settings["reward_net"], observation_space, action_space)
                   for member_settings in reward_net_kwargs["members"]]
        reward_net_kwargs["members"] = members

    reward_net = REWARD_NET_REGISTRY[reward_net_cls_name](
        observation_space,
        action_space,
        **reward_net_kwargs
    )

    return reward_net


def create_wrapped_reward_net(reward_net_settings, observation_space, action_space):

    result_net = create_reward_net(reward_net_settings[0], observation_space, action_space)
    for wrapper_cls_settings in reward_net_settings[1:]:

        wrapper_cls_name = get_cls_name(wrapper_cls_settings)

        wrapper_kwargs = {}
        if isinstance(wrapper_cls_settings, dict):
            wrapper_kwargs = wrapper_cls_settings[wrapper_cls_name]

        result_net = REWARD_NET_WRAPPER_REGISTRY[wrapper_cls_name](
            result_net,
            **wrapper_kwargs
        )

    return result_net
