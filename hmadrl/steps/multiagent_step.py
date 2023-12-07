import argparse
from marllib import marl

from hmadrl.marllib_utils import make_env
from hmadrl.settings_utils import load_settings, load_tune_settings, import_user_code, get_save_settings


def run(settings):
    import_user_code(settings["code"])

    algo_settings = load_tune_settings(settings['multiagent']['algo']['args'])
    model_settings = load_tune_settings(settings['multiagent']['model'])

    local_dir, restore_path = get_save_settings(settings["save"]["multiagent"])

    env_settings = settings['env']
    env_settings["step"] = "multiagent"
    env = make_env(env_settings)
    algo = marl._Algo(settings['multiagent']['algo']['name'])(hyperparam_source="common", **algo_settings)
    model = marl.build_model(env, algo, model_settings)
    algo.fit(env, model,
             share_policy='individual',
             stop={'timesteps_total': settings['multiagent']['timesteps']},
             local_dir=local_dir,
             restore_path=restore_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
    parser.add_argument('--settings', default='hmadrl.yaml', type=str,
                        help='path to settings file (default: hmadrl.yaml)')
    args = parser.parse_args()
    settings = load_settings(args.settings)
    run(settings)