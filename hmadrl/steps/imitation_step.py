import argparse
import pathlib
from time import time

import minari
import numpy as np
import optuna
from marllib import marl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

import hmadrl
from hmadrl.MARLlibWrapper import GymnasiumFixedHorizon
from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.imitation_utils import make_trajectories, init_as_multiagent, \
    create_imitation_models_from_settings, get_inner_algo_class_from_settings
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, make_env
from hmadrl.presetted_agents_env import SingleAgent
from hmadrl.settings_utils import load_settings, load_optuna_settings, \
    import_user_code, get_save_dir


def run(settings):
    hmadrl.marllib_utils.STEP_NAME = "imitation"
    import_user_code(settings["code"])

    trajectories = minari.MinariDataset(settings['save']['trajectory'])
    trajectories = make_trajectories(trajectories)

    rng = np.random.default_rng(0)

    env_settings = settings['env']
    env = make_env(env_settings)
    env_instance, _ = env
    algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                              **settings['multiagent']['algo']['args'])
    model = marl.build_model(env, algo, settings['multiagent']['model'])
    trainer = load_trainer(algo, env, model, settings['save']['multiagent'])

    policy_mapping = create_policy_mapping(env_instance)
    policy_mapping = {agent_id: trainer.get_policy(policy_id) for agent_id, policy_id in policy_mapping.items()}
    human_agent = settings['rollout']['human_agent']
    human_policy = policy_mapping[human_agent]
    policy_mapping.pop(human_agent, None)

    def make_rollout_env():
        rollout_env, _ = make_env(env_settings)
        rollout_env = SingleAgent(rollout_env, policy_mapping, human_agent)
        rollout_env = GymnasiumFixedHorizon(rollout_env, 300)
        return rollout_env

    rollout_env = make_vec_env(make_rollout_env, n_envs=4)

    def objective(trial: optuna.Trial):

        optuna_settings = load_optuna_settings(settings['imitation'], trial)
        inner_algo, reward_net = create_imitation_models_from_settings(settings, rollout_env, optuna_settings)

        use_multiagent_init = settings['imitation']['inner_algo'].get('use_multiagent_init', False)
        if use_multiagent_init:
            init_as_multiagent(inner_algo.policy, human_policy)

        imitation_algo = settings["imitation"]["algo"]["name"]
        imitation_inner_algo = settings["imitation"].get("inner_algo", {}).get("name", "none")
        path = (f"{get_save_dir(settings['save']['human_model'])}/"
                f"{imitation_algo}-{imitation_inner_algo}-{int(time())}-{trial.number}")

        trainer = IMITATION_REGISTRY[optuna_settings['algo']['name']](rollout_env, trajectories, rng, inner_algo,
                                                                      reward_net,
                                                                      optuna_settings['algo'].get('args', {}), path)
        total_timesteps = settings['imitation']['timesteps']
        checkpoint_freq = settings['imitation'].get("checkpoint_freq", total_timesteps)

        def callback(timestep):
            if timestep % checkpoint_freq == 0:
                trainer.save(timestep)

        try:
            trainer.train(total_timesteps, callback=callback)
            trainer.save(total_timesteps)
        except Exception as e:
            print(e)
            raise optuna.exceptions.TrialPruned()

        policy, _ = trainer.load(str(pathlib.Path(path) / str(total_timesteps)), 'cpu', type(inner_algo))
        mean, std = evaluate_policy(policy, rollout_env)
        return mean

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=settings['imitation'].get('trials', 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learn humanoid agent. Third step of HMADRL algorithm.')
    parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                        help='path to settings file (default: hmadrl.yaml)')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run(settings)
