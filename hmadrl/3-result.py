import argparse
import json
import pathlib
from time import time

import numpy as np
import optuna
from marllib import marl
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.imitation_utils import make_trajectories, init_as_multiagent
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, make_env, find_latest_dir
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, create_inner_algo_from_settings, load_optuna_settings, \
    import_user_code, get_inner_algo_class_from_settings

parser = argparse.ArgumentParser(description='Learn humanoid agent. Third step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)
import_user_code(settings["code"])

trajectories = np.load(settings['save']['trajectory'])
trajectories = make_trajectories(trajectories)

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
rollout_env.render_mode = "human"

checkpoint_path = find_latest_dir(pathlib.Path(settings["save"]["human_model"]))
inner_algo_cls = get_inner_algo_class_from_settings(settings["imitation"])
assert inner_algo_cls is not None, "Specified inner_algo is not supported"

rng = np.random.default_rng(0)
trainer = IMITATION_REGISTRY[settings["imitation"]['algo']['name']]
policy = trainer.load(str(checkpoint_path / "model.zip"), 'cpu', inner_algo_cls)
mean, std = evaluate_policy(policy, rollout_env, render=True, n_eval_episodes=1)
print("Eval:", mean, std)
