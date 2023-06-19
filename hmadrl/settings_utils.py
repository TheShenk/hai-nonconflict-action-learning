from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def load_settings(path: str):
    with open(path, 'r') as settings_file:
        settings = load(settings_file, Loader=Loader)
    return settings