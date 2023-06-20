from time import time
from typing import Dict, Type

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.density import DensityAlgorithm
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.logger import configure
from imitation.util.networks import RunningNorm
from stable_baselines3 import PPO, SAC, A2C, DDPG, DQN, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy


class ImitationTrainer:

    def __init__(self, venv, demonstrations, rng, inner_algo, algo_args):
        self.venv = venv
        self.demonstrations = demonstrations
        self.rng = rng
        self.inner_algo = inner_algo
        self.algo_args = algo_args
        self.logger = configure(f"./tensorboard/imitation/{type(self).__name__}-{int(time())}", ('tensorboard',))

    @staticmethod
    def load(path, device, inner_algo):
        pass

    def save(self, path):
        pass

    def train(self, timesteps):
        pass


class BaseImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo, algo_args):
        super().__init__(venv, demonstrations, rng, inner_algo, algo_args)
        self.inner_algo = inner_algo
        self.trainer = algo(
            venv=venv,
            demonstrations=demonstrations,
            rl_algo=self.inner_algo,
            rng=rng,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(path, device, inner_algo_cls: Type[BasePolicy]):
        return inner_algo_cls.load(path=path, device=device)

    def save(self, path):
        self.inner_algo.save(path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class BCTrainer(ImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, algo_args):
        super().__init__(venv, demonstrations, rng, inner_algo, algo_args)
        self.trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=demonstrations,
            rng=rng,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(path, device, inner_algo=None):
        return bc.reconstruct_policy(path, device)

    def save(self, path):
        self.trainer.save_policy(path)

    def train(self, epochs):
        self.trainer.train(n_epochs=epochs)


class GenerativeAdversarialImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo: BasePolicy, algo_args):
        super().__init__(venv, demonstrations, rng, inner_algo, algo_args)
        # TODO: настройки RewardNet. Сложность - для них существуют свои оболочки, то есть не очень понятно как
        #  указать какой вид сети использовать.
        #  Указывать массив используемых оболочек?
        #  Определить в импортирумом python-файле?
        self.reward_net = BasicRewardNet(
            venv.observation_space,
            venv.action_space,
            normalize_input_layer=RunningNorm,
        )
        self.inner_algo = inner_algo
        self.trainer = algo(
            demonstrations=demonstrations,
            venv=venv,
            gen_algo=self.inner_algo,
            reward_net=self.reward_net,
            custom_logger=self.logger,
            **algo_args
        )

    @staticmethod
    def load(path, device, inner_algo: BasePolicy):
        return inner_algo.load(path=path, device=device)

    def save(self, path):
        self.inner_algo.save(path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class GAILTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, algo_args):
        super().__init__(GAIL, venv, demonstrations, rng, inner_algo, algo_args)


class AIRLTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, algo_args):
        super().__init__(AIRL, venv, demonstrations, rng, inner_algo, algo_args)


class DensityTrainer(BaseImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo, algo_args):
        super().__init__(DensityAlgorithm, venv, demonstrations, rng, inner_algo, algo_args)

    def train(self, timesteps):
        self.trainer.train()


IMITATION_REGISTRY: Dict[str, Type[ImitationTrainer]] = {
    "bc": BCTrainer,
    "gail": GAILTrainer,
    "airl": AIRLTrainer,
    "density": DensityTrainer,
    # TODO: dagger (требует слияния 2 и 3 шага),
    #  MCE-IRL (поддерживает только TabularPOMDP среды из библиотеки seals)
    #  DI-engine
}

RL_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}
