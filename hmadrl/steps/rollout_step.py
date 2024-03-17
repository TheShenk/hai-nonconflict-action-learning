import gymnasium.envs.registration
import numpy as np
from marllib import marl
import minari
import argparse

import hmadrl
from hmadrl.MARLlibWrapper import GymnasiumFixedHorizon
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, rollout, make_env
from hmadrl.presetted_agents_env import SingleAgent
from hmadrl.settings_utils import load_settings, import_user_code


def run(settings):
    hmadrl.marllib_utils.STEP_NAME = "rollout"
    user = import_user_code(settings["code"])

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
    policy_mapping.pop(human_agent, None)

    rollout_env = SingleAgent(env_instance, policy_mapping, human_agent)
    rollout_env.spec = gymnasium.envs.registration.EnvSpec(id=settings['env']['name'])
    rollout_env = GymnasiumFixedHorizon(rollout_env, settings["rollout"].get("episode_timesteps", 300))
    rollout_env = minari.DataCollectorV0(rollout_env)

    reward = rollout(rollout_env, user.policy, settings['rollout']['episodes'])
    rollout_env.save_to_disk(settings['save']['trajectory'], {
        "dataset_id": f"{settings['env']['name']}-human-v0",
        "minari_version": "~=0.4.1"
    })
    rollout_env.close()
    print(f"Reward: {np.mean(reward)} +- {np.std(reward)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect human trajectories in environments with trained agents. '
                                                 'Second step of HMADRL algorithm.')
    parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                        help='path to settings file (default: hmadrl.yaml)')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run(settings)
