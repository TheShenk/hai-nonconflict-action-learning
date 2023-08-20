import pathlib
from typing import Dict, Type

import torch as th

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.density import DensityAlgorithm
# from imitation.algorithms.sqil import SQIL #TODO: when PyPi version will be updated
from imitation.policies.base import FeedForward32Policy, SAC1024Policy, ZeroPolicy, RandomPolicy
from imitation.util.logger import configure

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, SAC, A2C, DDPG, DQN, TD3
from sb3_contrib import TRPO, ARS, QRDQN, TQC


class ImitationTrainer:

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        self.venv = venv
        self.demonstrations = demonstrations
        self.rng = rng
        self.inner_algo = inner_algo
        self.reward_net = reward_net
        self.algo_args = algo_args
        self.path = pathlib.Path(path)
        self.model_path = str(self.path / "model.zip")
        self.reward_net_path = str(self.path / "reward_net.pt")
        self.logger = configure(self.path, ('tensorboard',))

    @staticmethod
    def load(path, device, inner_algo):
        pass

    def save(self):
        pass

    def train(self, timesteps):
        pass


class BaseImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)
        self.trainer = algo(
            venv=venv,
            demonstrations=demonstrations,
            rl_algo=self.inner_algo,
            rng=rng,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(checkpoint_path: str, device, inner_algo=None):
        checkpoint_path = pathlib.Path(checkpoint_path)
        return (inner_algo.load(path=str(checkpoint_path / "model.zip"), device=device),
                th.load(str(checkpoint_path / "reward_net.pt")))

    def save(self):
        th.save(self.reward_net, self.reward_net_path)
        self.inner_algo.save(self.model_path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class BCTrainer(ImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)
        self.trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=demonstrations,
            rng=rng,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(checkpoint_path: str, device, inner_algo=None):
        checkpoint_path = pathlib.Path(checkpoint_path)
        return (bc.reconstruct_policy(str(checkpoint_path / "model.zip"), device),
                th.load(str(checkpoint_path / "reward_net.pt")))

    def save(self):
        th.save(self.reward_net, self.reward_net_path)
        self.trainer.save_policy(self.model_path)

    def train(self, epochs):
        self.trainer.train(n_epochs=epochs)


class GenerativeAdversarialImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)
        self.trainer = algo(
            demonstrations=demonstrations,
            venv=venv,
            gen_algo=self.inner_algo,
            reward_net=self.reward_net,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(checkpoint_path: str, device, inner_algo=None):
        checkpoint_path = pathlib.Path(checkpoint_path)
        return (inner_algo.load(path=str(checkpoint_path / "model.zip"), device=device),
                th.load(str(checkpoint_path / "reward_net.pt")))

    def save(self):
        th.save(self.reward_net, self.reward_net_path)
        self.inner_algo.save(self.model_path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class GAILTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(GAIL, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)


class AIRLTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(AIRL, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)


class DensityTrainer(BaseImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(DensityAlgorithm, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)

    def train(self, timesteps):
        self.trainer.train()


class SQILTrainer(BaseImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(None, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path) # TODO: Fixme

    def train(self, timesteps):
        self.trainer.train()


IMITATION_REGISTRY: Dict[str, Type[ImitationTrainer]] = {
    "bc": BCTrainer,
    "gail": GAILTrainer,
    "airl": AIRLTrainer,
    # "density": DensityTrainer, #Incompatible becouse of using gym instead of gymnasium. Maybe will be fixed in future?
    # "sqil": SQILTrainer #TODO: when imitation version will be updated
    # TODO: dagger (требует слияния 2 и 3 шага),
    #  MCE-IRL (поддерживает только TabularPOMDP среды из библиотеки seals)
    #  DI-engine
}

RL_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    # "ars": ARS, # Incompatible because of evaluating policy before any actions. Conflicts with BufferingWrapper,
    # used by imitation
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "qr-dqn": QRDQN,
    "sac": SAC,
    "td3": TD3,
    "tqc": TQC,
    "trpo": TRPO
}
