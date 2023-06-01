import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib import marl

from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca

parser = argparse.ArgumentParser(description='Learning agent in environment. First step of HMADRL algorithm.')
parser.add_argument('--env', default='myfootball', type=str, nargs=1, help='name of environment (default: mpe)')
parser.add_argument('--map', default='hca', type=str, nargs=1, help='name of map (default: simple_spread)')
parser.add_argument('--algo', default='mappo', type=str, nargs=1, help='name of learning algorithm (default: mappo)')
parser.add_argument('--time', default=1000, type=int, nargs=1, help='number of timesteps (default: 1000)')

args = parser.parse_args()

env = marl.make_env(environment_name=args.env, map_name=args.map)
algo = marl._Algo(args.algo)(hyperparam_source="common")
model = marl.build_model(env, algo, {"core_arch": "mlp"})
algo.fit(env, model, stop={'timesteps_total': algo.time})

