import argparse
import pathlib

from marllib import marl

import hmadrl
from hmadrl.marllib_utils import make_env, find_checkpoint
from hmadrl.settings_utils import load_settings, load_tune_settings, import_user_code, get_save_settings

parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
parser.add_argument('--settings', default='hmadrl.yaml', type=str, help='path to settings file (default: hmadrl.yaml)')
args = parser.parse_args()
settings = load_settings(args.settings)

hmadrl.marllib_utils.STEP_NAME = "multiagent"
import_user_code(settings["code"])

algo_settings = load_tune_settings(settings['multiagent']['algo']['args'])
model_settings = load_tune_settings(settings['multiagent']['model'])

env_settings = settings['env']
env = make_env(env_settings)
algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common", **algo_settings)
model = marl.build_model(env, algo, model_settings)

local_dir, checkpoint = get_save_settings(settings['save']['multiagent'])
if not checkpoint["model_path"]:
    checkpoint_path = find_checkpoint(algo.name,
                                      env[1]['env_args']['map_name'],
                                      model[1]['model_arch_args']['core_arch'],
                                      local_dir)
    params_path = pathlib.Path(checkpoint_path).parent / '..' / 'params.json'
else:
    checkpoint_path = checkpoint["model_path"]
    params_path = checkpoint["params_path"]

local_dir = pathlib.Path(local_dir) / 'results'

algo.render(env, model,
            share_policy='individual',
            local_dir=str(local_dir),
            restore_path={
                'params_path': str(params_path),
                'model_path': str(checkpoint_path),
                'render': True
            })
