import imitation.data.rollout
import numpy as np
import pygame
from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl import recursive_dict_update, dict_update
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune import register_env
from ray.tune.utils import merge_dicts

from hmadrl.custom_policy import PyGamePolicy
from hmadrl.human_recorder import HumanRecorder
from hmadrl.presetted_agents_env import PreSettedAgentsEnv

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
        "evaluation_num_episodes": 1,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "record_env": True,
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
parser.add_argument('--env', default='myfootball', type=str, help='name of environment (default: myfootball)')
parser.add_argument('--map', default='hca', type=str, help='name of map (default: hca)')
parser.add_argument('--algo', default='mappo', type=str, help='name of learning algorithm (default: mappo)')
parser.add_argument('--episodes', default=5, type=int, help='number of episodes (default: 5)')
parser.add_argument('--checkpoint', type=str, help='path to checkpoint from first step')
parser.add_argument('--trajectory', type=str, help='path to file to save human actions and observations')

cli_args = parser.parse_args()


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

env = marl.make_env(environment_name=cli_args.env, map_name=cli_args.map)
algo = marl._Algo(cli_args.algo)(hyperparam_source="common")
model = marl.build_model(env, algo, {"core_arch": "mlp"})

env_instance, env_info = env
model_class, model_info = model

# policies = {
#     "human": PolicySpec(PyGamePolicy(human_policy)),
#     "policy_0": PolicySpec()
# }
policies = {'policy_0': PolicySpec(action_space=env_instance.action_space,
                                   observation_space=env_instance.observation_space),
            'policy_1': PolicySpec(action_space=env_instance.action_space,
                                   observation_space=env_instance.observation_space)}

policy_mapping_fn = lambda agent_id: {"player_0": "human", "player_1": "policy_0"}[agent_id]

exp_info = env_info
exp_info = recursive_dict_update(exp_info, model_info)
exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

exp_info['algorithm'] = cli_args.algo
exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies, policy_mapping_fn)

ModelCatalog.register_custom_model(
    "current_model", model_class)

trainer = MAPPOTrainer({"framework": "torch",
              "multiagent": {
                  "policies": policies,
              },
              "model": {
                  "custom_model": "current_model",
                  "custom_model_config": merge_dicts(exp_info, env_info),
              },
              })

trainer.restore(cli_args.checkpoint)


def rollout(env, policy, episodes_count):
    for episode_index in range(episodes_count):
        done = False
        observation = env.reset()
        while not done:
            action = policy(observation)
            observation, reward, done, info = env.step(action)
            rollout_env.render('human')
    env.close()


rollout_env = PreSettedAgentsEnv(HumanRecorder(env_instance, 'player_0', cli_args.trajectory),
                                 {'player_1': trainer.get_policy('policy_1')}, 'player_0')
rollout_policy = PyGamePolicy(human_policy)
rollout(rollout_env, rollout_policy, cli_args.episodes)

# algo_runner = POlICY_REGISTRY[cli_args.algo]
# result = algo_runner(model_class, exp_info, run_config, env_info, stop_config, None)