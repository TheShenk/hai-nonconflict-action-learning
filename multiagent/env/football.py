import random

import gym.spaces
import numpy as np
from gym import spaces

from agents.random_agent import RandomAgent
from gym_futbol.envs_v1 import Futbol
from gym_futbol.envs_v1.futbol_env import inverse_physic_vector_by_x_axis, WIDTH, HEIGHT, PLAYER_RADIUS, BALL_RADIUS, \
    TOTAL_TIME, NUMBER_OF_PLAYER, TIME_STEP


class TwoSideFootball(Futbol):

    def __init__(self, width=WIDTH, height=HEIGHT, player_radius=PLAYER_RADIUS, ball_radius=BALL_RADIUS,
                 total_time=TOTAL_TIME, debug=False,
                 number_of_player=NUMBER_OF_PLAYER, team_B_model=RandomAgent,
                 action_space_type="box", random_position=False,
                 team_reward_coeff=10, ball_reward_coeff=10, message_dims_number=0,
                 is_out_rule_enabled=True):
        super().__init__(width, height, player_radius, ball_radius, total_time, debug, number_of_player, team_B_model,
                         action_space_type, random_position, team_reward_coeff, ball_reward_coeff, message_dims_number,
                         is_out_rule_enabled)

        self.ball_owner_side = random.choice(["left", "right"])
        self.out = None
        self.done = None
        self.ball_init = None
        self.init_distance = {}
        self.goal_position = {
            self.team_A: [self.width, self.height/2],
            self.team_B: [0, self.height/2]
        }

    def _step(self, team, action, action_space_type):
        team_action = self.map_action_to_players(action, action_space_type)
        self.init_distance[team] = self._ball_to_team_distance_arr(team)
        self._process_team_action(team.player_array, team_action, action_space_type)

    def act(self, team_A_action):
        self._step(self.team_A, team_A_action, self.action_space_type[0])

    def inverted_act(self, team_B_action):

        if self.action_space_type[1] == "box":
            for action in team_B_action:
                inverse_physic_vector_by_x_axis(action)

        self._step(self.team_B, team_B_action, self.action_space_type[1])

    def calculate_reward(self, team):
        reward = 0

        # get reward
        if not self.out:
            ball_after = self.ball.get_position()

            reward += self.get_team_reward(self.init_distance[team], team)
            reward += self.get_ball_reward(self.ball_init, ball_after, self.goal_position[team])

        if self.ball_contact_goal():
            bx, _ = self.ball.get_position()

            goal_reward = 1000
            is_in_left_goal = bx > self.width // 2

            if team == self.team_A:
                reward += goal_reward if is_in_left_goal else -goal_reward
            else:
                reward += -goal_reward if is_in_left_goal else goal_reward

        return reward

    def calculate_left_reward(self):
        return self.calculate_reward(self.team_A)

    def calculate_right_reward(self):
        return self.calculate_reward(self.team_B)

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

    # ????????????????????????, ?????????? ???????????????????? Monitor-wrapper
    def step(self, team_A_action):
        reward = np.array([self.calculate_left_reward(), self.calculate_right_reward()])
        return self.observation, reward, self.done, {}