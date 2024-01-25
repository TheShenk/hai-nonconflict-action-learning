import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('files', type=str, nargs='+')
parser.add_argument('--output', type=str)
args = parser.parse_args()

episode_id = 0
total_steps = 0
with h5py.File(args.output, 'w') as result_file:
    for file in args.files:
        with h5py.File(file, 'r') as data:
            for episode in data:
                steps_count = len(data[episode]["observations"])
                if steps_count > 10:
                    data.copy(episode, result_file, name=f'episode_{episode_id}')
                    episode_id += 1
                    total_steps += steps_count
            result_file.attrs.update(data.attrs)
    result_file.attrs['total_episodes'] = episode_id
    result_file.attrs['total_steps'] = total_steps
    print(f"Episodes: {episode_id}, Steps: {total_steps}")

