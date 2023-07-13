from marllib import marl
import argparse

from hmadrl.marllib_utils import load_trainer, create_policy_mapping, rollout
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, import_user_code

parser = argparse.ArgumentParser(description='Collect human trajectories. Second step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                    help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)
user = import_user_code(settings["code"])

env = marl.make_env(environment_name=settings['env']['name'],
                    map_name=settings['env']['map'],
                    **settings['env']['args'])
env_instance, _ = env
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common",
                                                          **settings['multiagent']['algo']['args'])
model = marl.build_model(env, algo, settings['multiagent']['model'])
trainer = load_trainer(algo, env, model, settings['save']['retraining_model'])

policy_mapping = create_policy_mapping(env_instance)
policy_mapping = {agent_id: trainer.get_policy(policy_id) for agent_id, policy_id in policy_mapping.items()}

human_agent = settings['rollout']['human_agent']
policy_mapping.pop(human_agent, None)

rollout_env = PreSettedAgentsEnv(env_instance, policy_mapping, human_agent)

average_reward = rollout(rollout_env, user.policy, settings['result']['episodes'])
print("Average:", average_reward)
