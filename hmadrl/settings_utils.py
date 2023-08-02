import optuna
from ray import tune
from yaml import load, dump

from hmadrl.imitation_registry import RL_REGISTRY

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_settings(path: str):
    with open(path, 'r') as settings_file:
        settings = load(settings_file, Loader=Loader)
    return settings


def create_inner_algo_from_settings(rollout_env, settings):
    inner_algo_settings = settings.get('inner_algo', None)
    if inner_algo_settings:
        inner_algo_name = inner_algo_settings.get('name', None)
        if inner_algo_name:
            inner_algo_cls = RL_REGISTRY[inner_algo_name]
            inner_algo_args = inner_algo_settings.get('args', {})
            return inner_algo_cls(env=rollout_env, **inner_algo_args)
    return None


def get_inner_algo_class_from_settings(settings):
    inner_algo_settings = settings.get('inner_algo', None)
    if inner_algo_settings:
        inner_algo_name = inner_algo_settings.get('name', None)
        if inner_algo_name:
            return RL_REGISTRY[inner_algo_name]
    return None


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
