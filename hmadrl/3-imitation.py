import argparse
from time import time

import numpy as np
import optuna
from imitation.data.types import TrajectoryWithRew
from marllib import marl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, make_env
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, create_inner_algo_from_settings, load_optuna_settings, import_user_code


def make_trajectories(actions, observations, rewards, dones: np.ndarray):
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


parser = argparse.ArgumentParser(description='Learn humanoid agent. Third step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)
import_user_code(settings["code"])

trajectories = np.load(settings['save']['trajectory'])
actions = trajectories['actions']
observations = trajectories['observations']
rewards = trajectories['rewards']
dones = trajectories['dones']

trajectories = make_trajectories(actions, observations, rewards, dones)
rng = np.random.default_rng(0)

env = make_env(settings['env'])
env_instance, _ = env
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                          **settings['multiagent']['algo']['args'])
model = marl.build_model(env, algo, settings['multiagent']['model'])
trainer = load_trainer(algo, env, model, settings['save']['multiagent_model'])

policy_mapping = create_policy_mapping(env_instance)
policy_mapping = {agent_id: trainer.get_policy(policy_id) for agent_id, policy_id in policy_mapping.items()}
human_agent = settings['rollout']['human_agent']
policy_mapping.pop(human_agent, None)

rollout_env = PreSettedAgentsEnv(env_instance, policy_mapping, human_agent)
rollout_env = make_vec_env(lambda: rollout_env, n_envs=1)


def objective(trial: optuna.Trial):

    optuna_settings = load_optuna_settings(settings['imitation'], trial)
    inner_algo = create_inner_algo_from_settings(rollout_env, optuna_settings)
    path = f"{settings['save']['human_model']}/{optuna_settings['algo']['name']}-{int(time())}-{trial.number}"

    trainer = IMITATION_REGISTRY[optuna_settings['algo']['name']](rollout_env, trajectories, rng, inner_algo,
                                                                  optuna_settings['algo']['args'], path)
    trainer.train(settings['imitation']['timesteps'])
    trainer.save()

    policy = trainer.load(f"{path}/model.zip", 'cpu', type(inner_algo))
    mean, std = evaluate_policy(policy, rollout_env)
    return mean


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=settings['imitation'].get('trials', 1))
