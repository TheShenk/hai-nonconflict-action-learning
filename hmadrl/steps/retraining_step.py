import json
import pathlib

from marllib import marl
import argparse

from marllib.marl import POlICY_REGISTRY, recursive_dict_update
from ray.rllib.policy.policy import PolicySpec

import hmadrl
from hmadrl.custom_policy import ImitationPolicy, HumanPolicy
from hmadrl.imitation_registry import IMITATION_REGISTRY, RL_REGISTRY
from hmadrl.imitation_utils import find_imitation_checkpoint
from hmadrl.marllib_utils import find_checkpoint, create_policy_mapping, get_config, find_latest_dir, make_env
from hmadrl.settings_utils import load_settings, import_user_code, get_save_settings, get_save_dir


def run(settings):
    hmadrl.marllib_utils.STEP_NAME = "retraining"
    user = import_user_code(settings["code"])

    local_dir, restore_path = get_save_settings(settings["save"]["multiagent"])
    model_path, params_path = restore_path["model_path"], restore_path["params_path"]

    if not model_path:
        checkpoint_path = find_checkpoint(settings['multiagent']['algo']['name'],
                                          settings['env']['map'],
                                          settings['multiagent']['model']['core_arch'],
                                          local_dir)
        model_path = checkpoint_path
        params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'

    with open(params_path, 'r') as params_file:
        multiagent_params = json.load(params_file)

    experiment_path = find_imitation_checkpoint(settings)
    humanoid_model_path = str(experiment_path)

    inner_algo_cls = RL_REGISTRY[settings['imitation']['inner_algo']['name']]
    humanoid_model, _ = IMITATION_REGISTRY[settings['imitation']['algo']['name']].load(humanoid_model_path, 'cpu',
                                                                                       inner_algo_cls)

    env_settings = settings['env']
    env = make_env(env_settings)
    algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                              **multiagent_params['model']['custom_model_config'].get('algo_args', {}))
    model = marl.build_model(env, algo, multiagent_params['model']['custom_model_config']['model_arch_args'])

    env_instance, env_info = env
    model_class, model_info = model

    policies = {f'policy_{agent_num}': PolicySpec() for agent_num, agent_id in enumerate(env_instance.agents)}
    human_policy_index = env_instance.agents.index(settings["rollout"]["human_agent"])
    human_policy = f"policy_{human_policy_index}"
    policies[human_policy] = PolicySpec(ImitationPolicy(humanoid_model, model_class, len(env_instance.agents)))
    # policies["human"] = PolicySpec(HumanPolicy(user.policy))

    policy_mapping = create_policy_mapping(env_instance)
    policy_mapping[settings["rollout"]["human_agent"]] = human_policy

    policies_to_train = settings["retraining"].get("policies_to_train", env_instance.agents)
    policies_to_train = [policy_mapping[agent_id] for agent_id in policies_to_train]
    assert human_policy not in policies_to_train, "Don't specify human_agent in policies_to_train"

    def policy_mapping_fn(agent_id):
        return policy_mapping[agent_id]

    exp_info = env_info
    exp_info = recursive_dict_update(exp_info, model_info)
    exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

    exp_info['algorithm'] = settings['multiagent']['algo']['name']
    exp_info['restore_path'] = {
        "params_path": params_path,
        "model_path": model_path,
        'render': True
    }
    recursive_dict_update(exp_info, {
        "evaluation_interval": 1,
        "evaluation_num_episodes": 100,
        "evaluation_num_workers": 1,
        "evaluation_config": {
            "record_env": False,
            "render_env": True,
        }
    })
    exp_info["stop_timesteps"] = settings['retraining']['timesteps']
    exp_info['local_dir'] = settings['save']['retraining_model']
    exp_info, run_config, env_info, stop_config, restore_config = get_config(exp_info, env_instance, None, {
                "policies": policies,
                "policy_mapping_fn": policy_mapping_fn,
                "policies_to_train": policies_to_train
            })

    # recursive_dict_update(run_config, {
    #     "evaluation_interval": 1,
    #     "evaluation_num_episodes": 100,
    #     "evaluation_num_workers": 1,
    #     "evaluation_config": {
    #         "record_env": False,
    #         "render_env": True,
    #     }
    # })
    algo_runner = POlICY_REGISTRY[settings['multiagent']['algo']['name']]
    algo_runner(model_class, exp_info, run_config, env_info, stop_config, restore_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Retrain learned agents to play with human. Fourth step of HMADRL algorithm.')
    parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                        help='path to settings file (default: hmadrl.yaml)')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run(settings)
