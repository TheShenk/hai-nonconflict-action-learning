import gymnasium.envs.registration
from marllib import marl
import minari
import argparse

from hmadrl.marllib_utils import load_trainer, create_policy_mapping, rollout, make_env
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, import_user_code


def run(settings):
    user = import_user_code(settings["code"])

    env = make_env(settings['env'])
    env_instance, _ = env
    algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                              **settings['multiagent']['algo']['args'])
    model = marl.build_model(env, algo, settings['multiagent']['model'])
    trainer = load_trainer(algo, env, model, settings['save']['multiagent'])

    policy_mapping = create_policy_mapping(env_instance)
    policy_mapping = {agent_id: trainer.get_policy(policy_id) for agent_id, policy_id in policy_mapping.items()}

    human_agent = settings['rollout']['human_agent']
    policy_mapping.pop(human_agent, None)

    rollout_env = PreSettedAgentsEnv(env_instance, policy_mapping, human_agent)
    rollout_env.spec = gymnasium.envs.registration.EnvSpec(id=settings['env']['name'])
    rollout_env = minari.DataCollectorV0(rollout_env)

    average_reward = rollout(rollout_env, user.policy, settings['rollout']['episodes'])
    rollout_env.save_to_disk(settings['save']['trajectory'])
    rollout_env.close()
    print("Average:", average_reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
    parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                        help='path to settings file (default: hmadrl.yaml)')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run(settings)
