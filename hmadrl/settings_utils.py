import pathlib

import optuna
from ray import tune
from yaml import load

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_settings(path: str):
    with open(path, 'r') as settings_file:
        settings = load(settings_file, Loader=Loader)
    return settings


def import_user_code(filepath):
    import importlib.util

    spec = importlib.util.spec_from_file_location('human_policy_module', filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RAY_TUNE_REGISTER = {
    'grid_search': tune.grid_search,
    'choice': tune.choice,
    'uniform': tune.uniform,
    'randn': tune.randn,
    'loguniform': tune.loguniform
}

OPTUNA_REGISTER = {
    'categorical': optuna.Trial.suggest_categorical,
    'uniform': optuna.Trial.suggest_uniform,
    'float': optuna.Trial.suggest_float,
    'loguniform': optuna.Trial.suggest_loguniform,
    'int': optuna.Trial.suggest_int,
    'discrete_uniform': optuna.Trial.suggest_discrete_uniform
}


def get_args_kwargs(value):
    if type(value) == dict:
        return value.get('args', []), value.get('kwargs', {})
    else:
        return value, {}


def load_tune_settings(settings: dict):

    tune_settings = {}
    for key, value in settings.items():
        if key in RAY_TUNE_REGISTER.keys():
            args, kwargs = get_args_kwargs(value)
            return RAY_TUNE_REGISTER[key](*args, **kwargs)
        if type(value) == dict:
            tune_settings[key] = load_tune_settings(value)
        else:
            tune_settings[key] = value

    return tune_settings


def load_optuna_settings(settings: dict, trial: optuna.Trial, name='imitation'):

    optuna_settings = {}
    for key, value in settings.items():
        if key in OPTUNA_REGISTER.keys():
            args, kwargs = get_args_kwargs(value)
            return OPTUNA_REGISTER[key](trial, name, *args, **kwargs)
        if type(value) == dict:
            optuna_settings[key] = load_optuna_settings(value, trial, name=key)
        else:
            optuna_settings[key] = value

    return optuna_settings


def find_checkpoint_in_dir(checkpoint_dir: pathlib.Path):
    checkpoint = [item for item in checkpoint_dir.iterdir() if not item.name.startswith('.') and not item.suffixes]
    assert len(checkpoint) == 1, checkpoint
    return checkpoint[0]


def get_save_settings(settings):

    if isinstance(settings, dict):
        local_dir = settings["dir"]
        checkpoint_dir = pathlib.Path(settings["checkpoint"])

        model_path = find_checkpoint_in_dir(checkpoint_dir)
        params_path = checkpoint_dir / ".." / "params.json"

        return settings["dir"], dict(model_path=str(model_path), params_path=str(params_path))
    else:
        return settings, dict(model_path="", params_path="")


def get_save_dir(settings):
    if isinstance(settings, dict):
        return settings["dir"]
    else:
        return settings
