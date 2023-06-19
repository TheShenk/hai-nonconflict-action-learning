import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib import marl

from hmadrl.settings_utils import load_settings
from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca

parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, nargs=1, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

env = marl.make_env(environment_name=settings['env']['name'], map_name=settings['env']['map'])
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common")
model = marl.build_model(env, algo, {"core_arch": "mlp"})
algo.fit(env, model, stop={'timesteps_total': settings['multiagent']['timesteps']})

