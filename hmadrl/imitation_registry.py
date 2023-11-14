import pathlib
from typing import Dict, Type

import stable_baselines3.common.off_policy_algorithm
import torch as th

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL, SQILReplayBuffer
from imitation.util.logger import configure
from imitation.util import util

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
        return bc.reconstruct_policy(str(checkpoint_path / "model.zip"), device), None

    def save(self):
        util.save_policy(self.trainer.policy, self.model_path)

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


class SQILTrainer(ImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)
        self.replay_buffer = SQILReplayBuffer(
            inner_algo.buffer_size,
            inner_algo.observation_space,
            inner_algo.action_space,
            device=inner_algo.device,
            n_envs=inner_algo.n_envs,
            optimize_memory_usage=inner_algo.optimize_memory_usage,
            demonstrations=demonstrations
        )
        self.trainer = SQIL(venv=venv,
                            policy=algo_args.get("policy", "MlpPolicy"),
                            demonstrations=demonstrations,
                            rl_algo_class=type(inner_algo),
                            **algo_args)
        inner_algo.replay_buffer = self.replay_buffer
        self.trainer.rl_algo = inner_algo

    @staticmethod
    def load(checkpoint_path: str, device, inner_algo=None):
        checkpoint_path = pathlib.Path(checkpoint_path)
        return inner_algo.load(path=str(checkpoint_path / "model.zip"),
                               device=device), None

    def save(self):
        self.trainer.rl_algo.save(self.model_path)

    def train(self, timesteps):
        self.trainer.train(total_timesteps=timesteps)


IMITATION_REGISTRY: Dict[str, Type[ImitationTrainer]] = {
    "bc": BCTrainer,
    "gail": GAILTrainer,
    "airl": AIRLTrainer,
    "density": DensityTrainer,
    "sqil": SQILTrainer
    # TODO: dagger (требует слияния 2 и 3 шага),
    #  MCE-IRL (поддерживает только TabularPOMDP среды из библиотеки seals)
}

RL_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    # "ars": ARS, # Incompatible because of evaluating policy before any actions. Conflicts with BufferingWrapper,
    # used by imitation
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "qr-dqn": QRDQN,
    # "sac": SAC, # Update torch to 1.13 to fix
    "td3": TD3,
    "tqc": TQC,
    "trpo": TRPO
}
