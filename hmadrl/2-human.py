import pathlib

import numpy as np
import pygame
from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl import recursive_dict_update, POlICY_REGISTRY, dict_update
from ray.rllib.policy.policy import PolicySpec

from hmadrl.custom_policy import PyGamePolicy

from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca

def get_cc_config(exp_info, env, stop, policies, policy_mapping_fn):

    env_info = env.get_env_info()
    agent_name_ls = env.agents
    env_info["agent_name_ls"] = agent_name_ls
    env.close()

    run_config = {
        "seed": int(exp_info["seed"]),
        "env": exp_info["env"] + "_" + exp_info["env_args"]["map_name"],
        "num_gpus_per_worker": exp_info["num_gpus_per_worker"],
        "num_gpus": exp_info["num_gpus"],
        "num_workers": exp_info["num_workers"],
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_0"]
        },
        "framework": exp_info["framework"],
        "evaluation_interval": exp_info["evaluation_interval"],
        "simple_optimizer": False  # force using better optimizer
    }

    stop_config = {
        "episode_reward_mean": exp_info["stop_reward"],
        "timesteps_total": exp_info["stop_timesteps"],
        "training_iteration": exp_info["stop_iters"],
    }

    stop_config = dict_update(stop_config, stop)

    restore_config = exp_info['restore_path']
    render_config = {
        "evaluation_interval": 1,
        "evaluation_num_episodes": 100,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "record_env": False,
            "render_env": True,
        }
    }

    run_config = recursive_dict_update(run_config, render_config)

    render_stop_config = {
        "training_iteration": 1,
    }

    stop_config = recursive_dict_update(stop_config, render_stop_config)

    return exp_info, run_config, env_info, stop_config, restore_config


parser = argparse.ArgumentParser(description='Collect human trajectories. Second step of HMADRL algorithm.')
parser.add_argument('--env', default='myfootball', type=str, help='name of environment (default: mpe)')
parser.add_argument('--map', default='hca', type=str, help='name of map (default: simple_spread)')
parser.add_argument('--algo', default='mappo', type=str, help='name of learning algorithm (default: mappo)')
parser.add_argument('--time', default=1000, type=int, help='number of timesteps (default: 1000)')
parser.add_argument('--checkpoint', type=pathlib.Path, help='path to checkpoint from first step')

args = parser.parse_args()

params_path = args.checkpoint / '..' / 'params.jsom'
model_candidates = [file for file in args.checkpoint.iterdir() if (len(file.suffix) == 0 and file.name[0] != '.')]
assert len(model_candidates) == 1, model_candidates
model_path = model_candidates[0]

def human_policy(key, obs):
    enemy_goal_position = np.array([1, 0])
    ball_position = obs[:2]
    ball_velocity = obs[2:4]

    ball_enemy_goal_vector = enemy_goal_position - ball_position
    ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

    move_direction = [0, 0]
    hit_direction = ball_enemy_goal_vector / ball_enemy_goal_distance

    if key == pygame.K_RIGHT or key == pygame.K_d:
        move_direction = [1, 0]
    elif key == pygame.K_DOWN or key == pygame.K_s:
        move_direction = [0, 1]
    elif key == pygame.K_LEFT or key == pygame.K_a:
        move_direction = [-1, 0]
    elif key == pygame.K_UP or key == pygame.K_w:
        move_direction = [0, -1]
    elif key == pygame.K_SPACE:
        hit_direction = [1, 0]

    return np.append(move_direction, hit_direction)

policies = {
    "human": PolicySpec(PyGamePolicy(human_policy)),
    "policy_0": PolicySpec()
}
policy_mapping_fn = lambda agent_id: {"player_0": "human", "player_1": "policy_0"}[agent_id]

env = marl.make_env(environment_name=args.env, map_name=args.map, render_env=True)
algo = marl._Algo(args.algo)(hyperparam_source="common")
model = marl.build_model(env, algo, {"core_arch": "mlp"})

env_instance, env_info = env
model_class, model_info = model

exp_info = env_info
exp_info = recursive_dict_update(exp_info, model_info)
exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

exp_info['algorithm'] = args.algo
exp_info['restore_path'] = {
    "params_path": params_path,
    "model_path": model_path
}
exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies, policy_mapping_fn)

algo_runner = POlICY_REGISTRY[args.algo]
result = algo_runner(model_class, exp_info, run_config, env_info, stop_config, None)