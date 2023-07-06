import argparse

from marllib import marl
from marllib.envs.base_env import ENV_REGISTRY
from marllib.envs.global_reward_env import COOP_ENV_REGISTRY

from hmadrl.settings_utils import load_settings, load_tune_settings
from multiagent.env.ray_football import create_ma_football_hca

ENV_REGISTRY["myfootball"] = create_ma_football_hca
COOP_ENV_REGISTRY["myfootball"] = create_ma_football_hca

parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

algo_settings = load_tune_settings(settings['multiagent']['algo']['args'])
model_settings = load_tune_settings(settings['multiagent']['model'])

env = marl.make_env(environment_name=settings['env']['name'], map_name=settings['env']['map'], **settings['env']['args'])
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common", **algo_settings)
model = marl.build_model(env, algo, model_settings)
algo.fit(env, model,
         share_policy='individual',
         stop={'timesteps_total': settings['multiagent']['timesteps']},
         local_dir=settings['save']['multiagent_model'])

