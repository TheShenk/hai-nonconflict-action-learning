import cloudpickle
from marllib import marl
from marllib.marl import recursive_dict_update, dict_update
from marllib.marl.algos.core.CC.mappo import MAPPOTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.util.ml_utils.dict import merge_dicts
from typing import Dict


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

class FakeEnv:

    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space


def load_trainer(algo_name, env, checkpoint_path):
    with open(checkpoint_path, 'rb') as checkpoint_file:
        checkpoint = cloudpickle.load(checkpoint_file)
    worker = cloudpickle.loads(checkpoint['worker'])
    policies: Dict[str, PolicySpec] = worker['policy_specs']

    algo = marl._Algo(algo_name)(hyperparam_source="common")
    model = marl.build_model(env, algo, {"core_arch": "mlp"})

    env_instance, env_info = env
    model_class, model_info = model

    exp_info = env_info
    exp_info = recursive_dict_update(exp_info, model_info)
    exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

    exp_info['algorithm'] = algo_name
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
