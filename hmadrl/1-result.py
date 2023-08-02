import argparse
import pathlib

from marllib import marl

from hmadrl.marllib_utils import make_env, find_checkpoint
from hmadrl.settings_utils import load_settings, load_tune_settings, import_user_code

parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)
import_user_code(settings["code"])

algo_settings = load_tune_settings(settings['multiagent']['algo']['args'])
model_settings = load_tune_settings(settings['multiagent']['model'])

env = make_env(settings['env'])
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common", **algo_settings)
model = marl.build_model(env, algo, model_settings)

checkpoint_path = find_checkpoint(algo.name,
                                  env[1]['env_args']['map_name'],
                                  model[1]['model_arch_args']['core_arch'],
                                  settings['save']['multiagent_model'])
params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'
local_dir = pathlib.Path(settings['save']['multiagent_model']) / 'results'

algo.render(env, model,
            share_policy='individual',
            local_dir=str(local_dir),
            restore_path={
                'params_path': str(params_path),
                'model_path': str(checkpoint_path)
            })
