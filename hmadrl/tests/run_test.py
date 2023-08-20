import pytest
import ray

from hmadrl.imitation_registry import IMITATION_REGISTRY, RL_REGISTRY
from hmadrl.steps.multiagent_step import run as run_step1
from hmadrl.steps.rollout_step import run as run_step2
from hmadrl.steps.imitation_step import run as run_step3
from hmadrl.steps.retraining_step import run as run_step4


MULTIAGENT_ALGORITHMS = ['ippo', 'mappo', 'vdppo', 'happo', 'itrpo', 'matrpo', 'hatrpo', 'ia2c', 'maa2c', 'coma',
                         'vda2c', 'iddpg']

IMITATION_ALGORITHMS = list(IMITATION_REGISTRY.keys())
RL_ALGORITHMS = list(RL_REGISTRY.keys())

ALL_STEPS_ALGORITHMS = ([(algo, "gail", "ppo") for algo in MULTIAGENT_ALGORITHMS] +
                        [("mappo", algo, "ppo") for algo in IMITATION_ALGORITHMS] +
                        [("mappo", "gail", algo) for algo in RL_ALGORITHMS])


def create_settings(multiagent_algo, imitation_algo, imitation_inner_algo):
    discrete = multiagent_algo == "coma" or imitation_inner_algo in {"dqn" or "qr-dqn"}
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
            "timesteps": 2048 if imitation_inner_algo not in {"ddpg", "td3", "sac", "tqc"} else 5,
            "algo": {
                "name": imitation_algo,
                "args": {} if imitation_algo in {"bc", "density"} else {
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
            "multiagent": "test_multiagent_model",
            "trajectory": "test_human_discrete.npz" if discrete else "test_human.npz",
            "human_model": "test_humanoid",
            "retraining_model": "test_retraining_model"
        },
        "code": "../../example/football-discrete.py" if discrete else "../../example/football.py"
    }


@pytest.mark.parametrize("algorithm", MULTIAGENT_ALGORITHMS)
def test_step1(algorithm):

    settings = create_settings(algorithm, None, None)
    try:
        run_step1(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("algorithm", MULTIAGENT_ALGORITHMS)
def test_step2(algorithm):
    settings = create_settings(algorithm, None, None)
    try:
        run_step2(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("multiagent_algorithm,imitation_algorithm,inner_algorithm", ALL_STEPS_ALGORITHMS)
def test_step3(multiagent_algorithm, imitation_algorithm, inner_algorithm):
    settings = create_settings(multiagent_algorithm, imitation_algorithm, inner_algorithm)
    try:
        run_step3(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)


@pytest.mark.parametrize("multiagent_algorithm,imitation_algorithm,inner_algorithm", ALL_STEPS_ALGORITHMS)
def test_step4(multiagent_algorithm, imitation_algorithm, inner_algorithm):
    settings = create_settings(multiagent_algorithm, imitation_algorithm, inner_algorithm)
    try:
        run_step4(settings)
    except Exception as e:
        if ray.is_initialized():
            ray.shutdown()
        pytest.fail(e)