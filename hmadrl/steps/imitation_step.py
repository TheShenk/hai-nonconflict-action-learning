import argparse
from time import time

import numpy as np
import optuna
import shimmy
from marllib import marl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.imitation_utils import make_trajectories, init_as_multiagent, create_wrapped_reward_net, \
    create_imitation_models_from_settings
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, make_env
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, load_optuna_settings, \
    import_user_code, get_save_dir


def run(settings):
    import_user_code(settings["code"])

    trajectories = np.load(settings['save']['trajectory'])
    trajectories = make_trajectories(trajectories)

    rng = np.random.default_rng(0)

    env = make_env(settings['env'])
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

    rollout_env = PreSettedAgentsEnv(env_instance, policy_mapping, human_agent)
    rollout_env = make_vec_env(lambda: rollout_env, n_envs=1)

    def objective(trial: optuna.Trial):

        optuna_settings = load_optuna_settings(settings['imitation'], trial)
        inner_algo, reward_net = create_imitation_models_from_settings(settings, rollout_env, optuna_settings)

        use_multiagent_init = settings['imitation']['inner_algo'].get('use_multiagent_init', False)
        if use_multiagent_init:
            init_as_multiagent(inner_algo.policy, human_policy)

        path = f"{get_save_dir(settings['save']['human_model'])}/{optuna_settings['algo']['name']}-{int(time())}-{trial.number}"

        trainer = IMITATION_REGISTRY[optuna_settings['algo']['name']](rollout_env, trajectories, rng, inner_algo,
                                                                      reward_net,
                                                                      optuna_settings['algo'].get('args', {}), path)
        trainer.train(settings['imitation']['timesteps'])
        trainer.save()

        policy, _ = trainer.load(path, 'cpu', type(inner_algo))
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
