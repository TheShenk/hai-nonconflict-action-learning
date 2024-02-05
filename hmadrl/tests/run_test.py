import pytest
import ray

from hmadrl.imitation_registry import IMITATION_REGISTRY, RL_REGISTRY
from hmadrl.steps.multiagent_step import run as run_step1
from hmadrl.steps.rollout_step import run as run_step2
from hmadrl.steps.imitation_step import run as run_step3
from hmadrl.steps.retraining_step import run as run_step4


MULTIAGENT_ALGORITHMS = ['itrpo', 'matrpo', 'hatrpo', 'ippo', 'mappo', 'vdppo', 'happo', 'ia2c', 'maa2c', 'coma',
                         'vda2c', 'iddpg']

IMITATION_ALGORITHMS = list(IMITATION_REGISTRY.keys())
RL_ALGORITHMS = list(RL_REGISTRY.keys())
ALL_STEPS_ALGORITHMS = list(set([(algo, "gail", "ppo") for algo in MULTIAGENT_ALGORITHMS] +
                                [("mappo", algo, "ddpg" if algo in {"sqil"} else "ppo") for algo in IMITATION_ALGORITHMS] +
                                [("mappo", "gail", algo) for algo in RL_ALGORITHMS]))


def create_settings(multiagent_algo, imitation_algo, imitation_inner_algo, discrete=False):
    discrete_name = 'discrete' if discrete else 'not_discrete'
    return {
        "env": {
            "name": "myfootball",
            "map": "full-discrete" if discrete else "full",
            "args": {},
            "mask_flag": False,
            "global_state_flag": False,
            "opp_action_in_cc": False,
            "agent_level_batch_update": True
        },
        "multiagent": {
            "timesteps": 100,
            "model": {
                "core_arch": "mlp",
                "encode_layer": "128-256"
            },
            "algo": {
                "name": multiagent_algo,
                "args": {}
            }
        },
        "rollout": {
            "episodes": 1,
            "human_agent": "red_0"
        },
        "imitation": {
            "timesteps": 5 if imitation_inner_algo in {"ddpg", "td3", "sac", "tqc"} or imitation_algo == "bc" else 2048,
            "algo": {
                "name": imitation_algo,
                "args": {} if imitation_algo in {"bc", "density", "sqil"} else {
                    "demo_batch_size": 128
                }
            },
            "inner_algo": {
                "name": imitation_inner_algo,
                "args": {
                    "policy": "MlpPolicy",
                    "device": "cpu"
                }
            }
        },
        "retraining": {
                "policies_to_train": ["red_1"],
                "timesteps": 1000
        },
        "save": {
            "multiagent": f"test_multiagent_model_{discrete_name}",
            "trajectory": f"test_human_{discrete_name}.npz",
            "human_model": f"test_humanoid_{discrete_name}",
            "retraining_model": f"test_retraining_model_{discrete_name}"
        },
        "code": "../example/football-discrete.py" if discrete else "../example/football.py"
    }


@pytest.mark.parametrize("algorithm", MULTIAGENT_ALGORITHMS + ['mappo-discrete'])
def test_step1(algorithm):

    settings = create_settings(algorithm if algorithm != 'mappo-discrete' else 'mappo',
                               None, None,
                               discrete=(algorithm in {'coma', 'mappo-discrete'}))
    try:
        run_step1(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("algorithm", MULTIAGENT_ALGORITHMS)
def test_step2(algorithm):
    settings = create_settings(algorithm, None, None,
                               discrete=algorithm == 'coma')
    try:
        run_step2(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("multiagent_algorithm,imitation_algorithm,inner_algorithm", ALL_STEPS_ALGORITHMS)
def test_step3(multiagent_algorithm, imitation_algorithm, inner_algorithm):
    settings = create_settings(multiagent_algorithm, imitation_algorithm, inner_algorithm,
                               discrete=(multiagent_algorithm == 'coma' or inner_algorithm in {"dqn", "qr-dqn"}))
    try:
        run_step3(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("multiagent_algorithm,imitation_algorithm,inner_algorithm", ALL_STEPS_ALGORITHMS)
def test_step4(multiagent_algorithm, imitation_algorithm, inner_algorithm):
    settings = create_settings(multiagent_algorithm, imitation_algorithm, inner_algorithm,
                               discrete=(multiagent_algorithm == 'coma' or inner_algorithm in {"dqn", "qr-dqn"}))
    try:
        run_step4(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)
