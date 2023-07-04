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
    imitation_settings = settings.get('imitation', None)
    if imitation_settings:
        inner_algo_settings = imitation_settings.get('inner_algo', None)
        if inner_algo_settings:
            inner_algo_name = inner_algo_settings.get('name', None)
            if inner_algo_name:
                inner_algo_cls = RL_REGISTRY[inner_algo_name]
                inner_algo_args = inner_algo_settings.get('args', {})
                return inner_algo_cls(env=rollout_env, **inner_algo_args)
    return None


def load_human_policy(filepath):
    import importlib.util

    spec = importlib.util.spec_from_file_location('human_policy_module', filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.policy