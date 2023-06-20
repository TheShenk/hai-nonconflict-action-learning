import os.path
import pathlib

import cloudpickle
from marllib.marl import recursive_dict_update, dict_update, _Algo
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from ray.rllib import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.util.ml_utils.dict import merge_dicts
from typing import Dict, Tuple, Any


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


def find_checkpoint(algo_name: str, map_name: str, core_arch: str, local_dir_path: str):
    local_dir = pathlib.Path(local_dir_path)
    local_dir = local_dir / f"{algo_name}_{core_arch}_{map_name}"
    local_subdirs = [item for item in local_dir.iterdir() if item.is_dir()]
    local_subdirs.sort(key=lambda subdir: os.path.getmtime(subdir))

    checkpoint_dir = local_subdirs[-1]
    checkpoint_subdirs = [item for item in checkpoint_dir.iterdir() if item.is_dir()]
    checkpoint_subdirs.sort(key=lambda subdir: os.path.getmtime(subdir))

    checkpoint = checkpoint_subdirs[-1]
    checkpoint = [item for item in checkpoint.iterdir() if not item.name.startswith('.') and not item.suffixes]
    assert len(checkpoint) == 1, checkpoint
    return str(checkpoint[0])


def load_trainer(algo: _Algo, env: Tuple[MultiAgentEnv, Dict], model: Tuple[Any, Dict], local_dir_path: str):
    env_instance, env_info = env
    model_class, model_info = model

    checkpoint_path = find_checkpoint(algo.name,
                                      env_info['env_args']['map_name'],
                                      model_info['model_arch_args']['core_arch'],
                                      local_dir_path)

    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint = cloudpickle.load(checkpoint_file)
    worker = cloudpickle.loads(checkpoint['worker'])
    print(worker.keys())
    policies: Dict[str, PolicySpec] = worker['policy_specs']

    exp_info = env_info
    exp_info = recursive_dict_update(exp_info, model_info)
    exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

    exp_info['algorithm'] = algo.name
    exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies,
                                                                                lambda: None)

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

    trainer.restore(checkpoint_path)
    return trainer


def create_policy_mapping(env: MultiAgentEnv) -> Dict[str, str]:
    policy_mapping = {agent_id: f"policy_{agent_number}" for agent_number, agent_id in enumerate(env.agents)}
    return policy_mapping
