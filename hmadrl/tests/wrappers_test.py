from pettingzoo.mpe import simple_spread_v3

from hmadrl.MARLlibWrapper import TimeLimit


def test_time_limit():

    time_limit = 10

    env = simple_spread_v3.parallel_env()
    env = TimeLimit(env, time_limit)

    env.reset()
    for _ in range(time_limit):
        _, _, _, truncated, _ = env.step({agent: env.action_space(agent).sample() for agent in env.agents})
        assert not any(truncated.values())
    _, _, _, truncated, _ = env.step({agent: env.action_space(agent).sample() for agent in env.agents})
    assert all(truncated.values())
