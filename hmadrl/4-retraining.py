import pathlib

from marllib import marl
import argparse

from marllib.envs.base_env import ENV_REGISTRY
from marllib.marl import recursive_dict_update, POlICY_REGISTRY
from ray.rllib.policy.policy import PolicySpec
from stable_baselines3 import PPO

from hmadrl.custom_policy import ImitationPolicy

from hmadrl.imitation_registry import IMITATION_REGISTRY
from hmadrl.marllib_utils import get_cc_config
from multiagent.env.ray_football import create_ma_football_hca
ENV_REGISTRY["myfootball"] = create_ma_football_hca


parser = argparse.ArgumentParser(description='Retraiin learned agents to play with human. Fourth step of HMADRL algorithm.')
parser.add_argument('--env', default='myfootball', type=str, help='name of environment (default: myfootball)')
parser.add_argument('--map', default='hca', type=str, help='name of map (default: hca)')
parser.add_argument('--algo', default='mappo', type=str, help='name of learning algorithm (default: mappo)')
parser.add_argument('--imit-algo', default='airl', type=str, help='name of imitation algorithm (default: airl)')
parser.add_argument('--time', default=1000, type=int, help='number of timesteps (default: 1000)')
parser.add_argument('--checkpoint', type=pathlib.Path, help='path to checkpoint from first step')
parser.add_argument('--human-model', type=pathlib.Path, help='path to file to load humanoid agent model from third step')

args = parser.parse_args()

params_path = args.checkpoint / '..' / 'params.jsom'
model_candidates = [file for file in args.checkpoint.iterdir() if (len(file.suffix) == 0 and file.name[0] != '.')]
assert len(model_candidates) == 1, model_candidates
model_path = model_candidates[0]

humanoid_model = IMITATION_REGISTRY[args.imit_algo].load(args.human_model, 'cpu', PPO)

policies = {
    "human": PolicySpec(ImitationPolicy(humanoid_model)),
    "policy_0": PolicySpec()
}
policy_mapping_fn = lambda agent_id: {"player_0": "human", "player_1": "policy_0"}[agent_id]

env = marl.make_env(environment_name=args.env, map_name=args.map)
algo = marl._Algo(args.algo)(hyperparam_source="common")
model = marl.build_model(env, algo, {"core_arch": "mlp"})

env_instance, env_info = env
model_class, model_info = model

exp_info = env_info
exp_info = recursive_dict_update(exp_info, model_info)
exp_info = recursive_dict_update(exp_info, algo.algo_parameters)

exp_info['algorithm'] = args.algo
exp_info['restore_path'] = {
    "params_path": params_path,
    "model_path": model_path
}
exp_info, run_config, env_info, stop_config, restore_config = get_cc_config(exp_info, env_instance, None, policies, policy_mapping_fn)

algo_runner = POlICY_REGISTRY[args.algo]
result = algo_runner(model_class, exp_info, run_config, env_info, stop_config, None)