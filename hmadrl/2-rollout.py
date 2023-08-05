from marllib import marl
import argparse

from hmadrl.human_recorder import HumanRecorder
from hmadrl.marllib_utils import load_trainer, create_policy_mapping, rollout, make_env
from hmadrl.presetted_agents_env import PreSettedAgentsEnv
from hmadrl.settings_utils import load_settings, import_user_code

parser = argparse.ArgumentParser(description='Collect human trajectories. Second step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                    help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)
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

rollout_env = PreSettedAgentsEnv(HumanRecorder(env_instance, human_agent, settings['save']['trajectory']),
                                 policy_mapping, human_agent)

average_reward = rollout(rollout_env, user.policy, settings['rollout']['episodes'])
print("Average:", average_reward)
