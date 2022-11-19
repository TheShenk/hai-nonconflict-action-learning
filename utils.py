import pymunk
from IPython import display
from matplotlib import pyplot as plt


def visualize_matplotlib(env, reward):
    padding = 5

    plt.clf()
    plt.title(f"Reward: {reward}", loc='left')
    ax = plt.axes(xlim=(0 - padding, env.width + padding), ylim=(0 - padding, env.height + padding))
    ax.set_aspect("equal")

    draw_options = pymunk.matplotlib_util.DrawOptions(ax)
    env.space.debug_draw(draw_options)
    display.display(plt.gcf())
    display.clear_output(wait=True)


def run_model(env, model, visualize):
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _states = model.predict(ob)
        ob, reward, done, info = env.step(action)
        visualize(env, reward)
        total_reward += reward
    return total_reward
