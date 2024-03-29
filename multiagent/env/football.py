import random

from gym_futbol.envs_v1 import Futbol
from gym_futbol.envs_v1.futbol_env import inverse_physic_vector_by_x_axis, TIME_STEP, get_vec


class TwoSideFootball(Futbol):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ball_owner_side = random.choice(["left", "right"])
        self.out = None
        self.done = None
        self.ball_init = None
        self.init_distance = {}
        self.goal_position = {
            'red': [self.width, self.height/2],
            'blue': [0, self.height/2]
        }
        self.teams = {
            'red': self.team_A,
            'blue': self.team_B
        }

    def _step(self, team_name, action, action_space_type):
        team = self.teams[team_name]
        team_action = self.map_action_to_players(action, action_space_type)
        self.init_distance[team_name] = self._ball_to_team_distance_arr(team)
        self._process_team_action(team.player_array, team_action, action_space_type)

    def act(self, team_A_action):
        self._step('red', team_A_action, self.action_space_type[0])

    def inverted_act(self, team_B_action):

        if self.action_space_type[1] == "box":
            for action in team_B_action:
                inverse_physic_vector_by_x_axis(action)

        self._step('blue', team_B_action, self.action_space_type[1])

    def calculate_reward(self, team_name):
        reward = 0

        # get reward
        if not self.out:
            ball_after = self.ball.get_position()

            reward += self.get_team_reward(self.init_distance[team_name], self.teams[team_name])
            reward += self.get_ball_reward(self.ball_init, ball_after, self.goal_position[team_name])

        if self.ball_contact_goal():
            bx, _ = self.ball.get_position()
            is_in_left_goal = bx > self.width // 2

            if team_name == 'red':
                reward += self.goal_reward if is_in_left_goal else -self.goal_reward
            else:
                reward += -self.goal_reward if is_in_left_goal else self.goal_reward

        return reward

    def calculate_left_reward(self):
        return self.calculate_reward('red')

    def calculate_right_reward(self):
        return self.calculate_reward('blue')

    def commit(self):
        self.done = False
        self.ball_init = self.ball.get_position()
        self.out = self.check_and_fix_out_bounds() if self.is_out_rule_enabled else False

        self.space.step(TIME_STEP)
        self.observation, self.inverse_obs = self._get_observation()

        if self.ball_contact_goal():
            self._position_to_initial()
            self.ball_owner_side = random.choice(["left", "right"])

        self.current_time += TIME_STEP
        if self.current_time > self.total_time:
            self.done = True

        return self.observation, self.inverse_obs, self.done, {}

    # Используется, чтобы поддержать Monitor-wrapper
    # При испольовании с MARLlib код должен быть закомментирован. Иначе ошибка сериализации в pickle
    # def step(self, team_A_action):
    #     left_reward = self.calculate_left_reward()
    #     right_reward = self.calculate_right_reward()
    #     return self.observation, left_reward, self.done, {"left_reward": left_reward, "right_reward": right_reward}


# Данная среда позволяет обучать одновременно атакующего и вратаря в игре друг против друга.
class AttackingVsGoalkeeper(TwoSideFootball):

    def __init__(self, goalkeeper_reward_coeff=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.goalkeeper_init_pos = self.get_goalkeeper().get_position()
        self.goalkeeper_reward_coeff = goalkeeper_reward_coeff
        self.goalkeeper_goal_pos = self.goal_position[self.team_A].copy()

        # Смещение позиции, к которой стремится вратарь. Необходимо из-за особенностей среды, иначе вратарь
        # слишком близко к воротам и мяч задевает ворота, даже если там стоит вратарь
        self.goalkeeper_goal_pos[0] -= 5

    def get_goalkeeper(self):
        return self.team_B.player_array[0]

    def get_goalkeeper_reward(self, b_goalkeeper_pos, a_goalkeeper_pos, goal_pos):
        _, before_distance_to_goal = get_vec(goal_pos, b_goalkeeper_pos)
        _, after_distance_to_goal = get_vec(goal_pos, a_goalkeeper_pos)
        return self.goalkeeper_reward_coeff * (before_distance_to_goal - after_distance_to_goal)

    def calculate_right_reward(self):
        reward = 0

        ball_position = self.ball.get_position()
        goalkeeper_pos = self.get_goalkeeper().get_position()

        # get reward
        if not self.out:
            reward += self.get_ball_reward(self.ball_init, ball_position, self.goal_position[self.team_B])
            reward += self.get_goalkeeper_reward(self.goalkeeper_init_pos, goalkeeper_pos, self.goalkeeper_goal_pos)

        self.goalkeeper_init_pos = goalkeeper_pos

        ball_on_right_side = ball_position[0] < self.width // 2
        if self.ball_contact_goal() and ball_on_right_side:
            reward -= self.goal_reward

        return reward