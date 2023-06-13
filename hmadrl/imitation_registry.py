from typing import Dict, Type

from imitation.algorithms.adversarial.airl import AIRL
from imitation.algorithms.adversarial.gail import GAIL
from imitation.algorithms import bc
from imitation.algorithms.density import DensityAlgorithm
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from stable_baselines3.common.policies import BasePolicy


class ImitationTrainer:

    def __init__(self, venv, demonstrations, rng, inner_algo):
        self.venv = venv
        self.demonstrations = demonstrations
        self.rng = rng
        self.inner_algo = inner_algo

    @staticmethod
    def load(path, device, inner_algo):
        pass

    def save(self, path):
        pass

    def train(self, timesteps):
        pass


class BaseImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo):
        super().__init__(venv, demonstrations, rng, inner_algo)
        self.inner_algo = inner_algo(env=venv, policy='MlpPolicy', device='cpu')
        self.trainer = algo(
            venv=venv,
            demonstrations=demonstrations,
            rl_algo=inner_algo(env=venv, policy='MlpPolicy', device='cpu'),
            rng=rng
        )

    @staticmethod
    def load(path, device, inner_algo: BasePolicy):
        return inner_algo.load(path=path, device=device)

    def save(self, path):
        self.inner_algo.save(path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class BCTrainer(ImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo):
        super().__init__(venv, demonstrations, rng, inner_algo)
        self.trainer = bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=demonstrations,
            rng=rng,
            # TODO: device as arg
            device='cpu'
        )

    @staticmethod
    def load(path, device, inner_algo=None):
        return bc.reconstruct_policy(path, device)

    def save(self, path):
        self.trainer.save_policy(path)

    def train(self, epochs):
        self.trainer.train(n_epochs=epochs)


class GenerativeAdversarialImitationTrainer(ImitationTrainer):
    def __init__(self, algo, venv, demonstrations, rng, inner_algo: Type[BasePolicy]):
        super().__init__(venv, demonstrations, rng, inner_algo)
        self.reward_net = BasicRewardNet(
            venv.observation_space,
            venv.action_space,
            normalize_input_layer=RunningNorm,
        )
        self.inner_algo = inner_algo(env=venv, policy='MlpPolicy', device='cpu')
        self.trainer = algo(
            demonstrations=demonstrations,
            venv=venv,
            demo_batch_size=1024,
            gen_replay_buffer_capacity=2048,
            n_disc_updates_per_round=4,
            gen_algo=self.inner_algo,
            reward_net=self.reward_net
        )

    @staticmethod
    def load(path, device, inner_algo: BasePolicy):
        return inner_algo.load(path=path, device=device)

    def save(self, path):
        self.inner_algo.save(path)

    def train(self, timesteps):
        self.trainer.train(timesteps)


class GAILTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo):
        super().__init__(GAIL, venv, demonstrations, rng, inner_algo)


class AIRLTrainer(GenerativeAdversarialImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo):
        super().__init__(AIRL, venv, demonstrations, rng, inner_algo)


class DensityTrainer(BaseImitationTrainer):

    def __init__(self, venv, demonstrations, rng, inner_algo):
        super().__init__(DensityAlgorithm, venv, demonstrations, rng, inner_algo)

    def train(self, timesteps):
        self.trainer.train()


IMITATION_REGISTRY: Dict[str, Type[ImitationTrainer]] = {
    "bc": BCTrainer,
    "gail": GAILTrainer,
    "airl": AIRLTrainer,
    "density": DensityTrainer
    #TODO: dagger, MCE IRL, DI-enginebc_creator
}