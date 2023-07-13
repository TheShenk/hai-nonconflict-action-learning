import numpy as np


def attacking(observation, player_index, player_obs_len, enemy_goal_position):
    ball_position = observation[:2]
    ball_velocity = observation[2:4]

    players_observation = observation[4:]
    attacking_obs_start = player_index * player_obs_len
    attacking_position = players_observation[attacking_obs_start:attacking_obs_start + 2]
    attacking_velocity = players_observation[attacking_obs_start + 2:attacking_obs_start + 4]

    to_ball_vector = ball_position - attacking_position
    to_ball_distance = np.linalg.norm(to_ball_vector)

    ball_enemy_goal_vector = enemy_goal_position - ball_position
    ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

    return np.append(
        to_ball_vector / to_ball_distance,
        ball_enemy_goal_vector / ball_enemy_goal_distance
    )


def goalkeeper(observation, player_index, player_obs_len, goal_position, enemy_goal_position):
    ball_position = observation[:2]
    ball_velocity = observation[2:4]
    future_ball_position = ball_position + ball_velocity * 0.5

    players_observation = observation[4:]
    goalkeeper_obs_start = player_index * player_obs_len
    goalkeeper_position = players_observation[goalkeeper_obs_start:goalkeeper_obs_start + 2]
    goalkeeper_velocity = players_observation[goalkeeper_obs_start + 2:goalkeeper_obs_start + 4]
    future_goalkeeper_position = goalkeeper_position + goalkeeper_velocity

    to_goal_vector = goal_position - goalkeeper_position
    to_goal_distance = np.linalg.norm(to_goal_vector)

    to_ball_vector = future_ball_position - goalkeeper_position
    to_ball_distance = np.linalg.norm(to_ball_vector)

    ball_enemy_goal_vector = enemy_goal_position - ball_position
    ball_enemy_goal_distance = np.linalg.norm(ball_enemy_goal_vector)

    if to_ball_distance < 0.45:
        return np.append(
            to_ball_vector / to_ball_distance,
            ball_enemy_goal_vector / ball_enemy_goal_distance
        )
    else:
        return np.append(to_goal_vector / to_goal_distance, [0, 0])


class AsPolicy:

    def __init__(self, func, *args, **kwargs):

        self.func = func
        self.args = args
        self.kwargs = kwargs

    def predict(self, observation, *args, **kwargs):
        return self.func(observation, *self.args, **self.kwargs)