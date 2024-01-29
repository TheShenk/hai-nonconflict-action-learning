import pathlib
from typing import Dict, Type

import torch as th

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.density import DensityAlgorithm
from imitation.algorithms.sqil import SQIL, SQILReplayBuffer
from imitation.util.logger import configure
from imitation.util import util

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import PPO, A2C, DDPG, DQN, TD3
from sb3_contrib import TRPO, QRDQN, TQC


class ImitationTrainer:

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        self.venv = venv
        self.demonstrations = demonstrations
        self.rng = rng
        self.inner_algo = inner_algo
        self.reward_net = reward_net
        self.algo_args = algo_args
        self.path = pathlib.Path(path)
        self.logger = configure(self.path, ('tensorboard',))
        self.timesteps = 0

    @staticmethod
    def load(path, device, inner_algo):
        pass

    def save(self, timesteps):
        pass

    def train(self, timesteps, callback):
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

    def save(self, timesteps):
        current_save_dir = self.path / str(timesteps)
        current_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(current_save_dir / "model.zip")
        reward_net_path = str(current_save_dir / "reward_net.pt")

        th.save(self.reward_net, reward_net_path)
        self.inner_algo.save(model_path)

    def train(self, timesteps, callback):
        super().train(timesteps, callback)
        self.trainer.train(timesteps, callback)


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

    def save(self, timesteps):
        current_save_dir = self.path / str(timesteps)
        current_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(current_save_dir / "model.zip")
        util.save_policy(self.trainer.policy, model_path)

    def train(self, batches, callback):
        super().train(batches, callback)

        class OnBatchEndCallback:
            counter: int

            def __init__(self, callback):
                self.counter = 0
                self.callback = callback

            def __call__(self, *args, **kwargs):
                self.counter += 1
                self.callback(self.counter)

        self.trainer.train(n_batches=batches, on_batch_end=OnBatchEndCallback(callback))


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

    def save(self, timesteps):
        current_save_dir = self.path / str(timesteps)
        current_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(current_save_dir / "model.zip")
        reward_net_path = str(current_save_dir / "reward_net.pt")

        th.save(self.reward_net, reward_net_path)
        self.inner_algo.save(model_path)

    def train(self, timesteps, callback):
        super().train(timesteps, callback)
        self.trainer.train(timesteps, callback=callback)


class GAILTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(GAIL, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)


class AIRLTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(AIRL, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)


class DensityTrainer(BaseImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path):
        super().__init__(DensityAlgorithm, venv, demonstrations, rng, inner_algo, reward_net, algo_args, path)

    def train(self, timesteps, callback):
        for i in range(timesteps):
            self.trainer.train()
            callback(timesteps)
        self.timesteps += timesteps


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

    def save(self, timesteps):
        current_save_dir = self.path / str(timesteps)
        current_save_dir.mkdir(parents=True, exist_ok=True)
        model_path = str(current_save_dir / "model.zip")
        self.trainer.rl_algo.save(model_path)

    def train(self, timesteps, callback):
        super().train(timesteps, callback)
        self.trainer.train(total_timesteps=timesteps)  # TODO: callback


IMITATION_REGISTRY: Dict[str, Type[ImitationTrainer]] = {
    "bc": BCTrainer,
    "gail": GAILTrainer,
    "airl": AIRLTrainer,
    "density": DensityTrainer,
    "sqil": SQILTrainer
    # dagger # Need to combine step-2 and step-3
    # MCE-IRL # Support only TabularPOMDP environments from seals library
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
