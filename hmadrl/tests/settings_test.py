import gymnasium as gym
from imitation.rewards.reward_nets import AddSTDRewardWrapper, RewardEnsemble, BasicRewardNet, CnnRewardNet, \
    BasicShapedRewardNet, NormalizedRewardNet, RewardNet
from imitation.util.networks import RunningNorm

from hmadrl.imitation_utils import create_wrapped_reward_net, register_reward_net, register_reward_net_wrapper

from yaml import safe_load


def test_reward_net_ensemble():

    reward_net_settings = """
    imitation:
        reward_net:
            - RewardEnsemble:
                members:
                    - reward_net:
                        - BasicRewardNet:
                            use_done: True
                    - reward_net:
                        - BasicShapedRewardNet
            - AddSTDRewardWrapper:
                default_alpha: 3.1
    """

    settings = safe_load(reward_net_settings)
    reward_net = create_wrapped_reward_net(settings["imitation"]["reward_net"],
                                   gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                                   gym.spaces.Box(-1.0, 1.0, shape=(1,)))

    assert isinstance(reward_net, AddSTDRewardWrapper)
    assert reward_net.default_alpha == 3.1
    assert isinstance(reward_net.base, RewardEnsemble)
    assert len(reward_net.base.members) == 2
    assert isinstance(reward_net.base.members[0], BasicRewardNet)
    assert reward_net.base.members[0].use_done
    assert isinstance(reward_net.base.members[1], BasicShapedRewardNet)


def test_reward_net_registry():

    class CustomRewardNet(BasicRewardNet): pass

    class CustomRewardNetWrapper(NormalizedRewardNet):

        def __init__(self, base: RewardNet):
            super().__init__(base, RunningNorm)

    register_reward_net("CustomRewardNet", CustomRewardNet)
    register_reward_net_wrapper("CustomRewardNetWrapper", CustomRewardNetWrapper)

    reward_net_settings = """
    imitation:
        reward_net:
            - CustomRewardNet
            - CustomRewardNetWrapper
    """

    settings = safe_load(reward_net_settings)
    reward_net = create_wrapped_reward_net(settings["imitation"]["reward_net"],
                                   gym.spaces.Box(-1.0, 1.0, shape=(1,)),
                                   gym.spaces.Box(-1.0, 1.0, shape=(1,)))

    assert isinstance(reward_net, CustomRewardNetWrapper)
    assert isinstance(reward_net.base, CustomRewardNet)
