from copy import copy

import pygame

import hagl
from agents.random_agent import RandomAgent
from hagl import Position, Velocity, T
from hagl.functions import Normalize
from hagl.proxy import HAGLPyMunk
from .player import Player
from .ball import Ball
from .team import Team

import numpy as np
import random
import math
import pymunk.matplotlib_util
import pymunk
import matplotlib.pyplot as plt

WIDTH = 105
HEIGHT = 68
GOAL_SIZE = 20

PLAYER_RADIUS = 1.5
BALL_RADIUS = 1

TOTAL_TIME = 30  # 30 s

TIME_STEP = 0.1  # 0.1 s

# player number each team, less than 10
NUMBER_OF_PLAYER = 5

BALL_MAX_VELOCITY = 25
PLAYER_MAX_VELOCITY = 10

BALL_WEIGHT = 10
PLAYER_WEIGHT = 20

PLAYER_FORCE_LIMIT = 40
BALL_FORCE_LIMIT = 120

padding = 3

PLAYER_MAX_POSITION = np.array([WIDTH + padding, HEIGHT])
PLAYER_MIN_POSITION = np.array([0 - padding, 0])

PLAYER_MAX_VELOCITY_arr = np.array([PLAYER_MAX_VELOCITY, PLAYER_MAX_VELOCITY])
PLAYER_MIN_VELOCITY_arr = np.array([-PLAYER_MAX_VELOCITY, -PLAYER_MAX_VELOCITY])

BALL_MAX_POSITION_arr = np.array([WIDTH, HEIGHT])
BALL_MIN_POSITION_arr = np.array([-0, -0])

BALL_MAX_VELOCITY_arr = np.array([BALL_MAX_VELOCITY, BALL_MAX_VELOCITY])
BALL_MIN_VELOCITY_arr = np.array([-BALL_MAX_VELOCITY, -BALL_MAX_VELOCITY])

TEAMS_COUNT = 2
PHYSIC_OBSERVATION_DIMS_NUMBER = 4
BALLS_COUNT = 1

# get the vector pointing from [coor2] to [coor1] and
# its magnitude
def get_vec(coor_t, coor_o):
    vec = [coor_t[0] - coor_o[0], coor_t[1] - coor_o[1]]
    vec_mag = math.sqrt(vec[0]**2 + vec[1]**2)
    return vec, vec_mag


def inverse_by_x(body, min_x, max_x):
    inverse_body = body.copy()
    # 2 * average_x - body.position.x = 2 * (max_x + min_x) / 2 - body.position.x = min_x + max_x - body.position.x
    inverse_body.position = (min_x + max_x - body.position.x, body.position.y)
    inverse_body.velocity = (-body.velocity.x, body.velocity.y)

    return inverse_body

class PlayerObservation:
    position = Normalize(Position, left=PLAYER_MIN_POSITION, right=PLAYER_MAX_POSITION)
    velocity = Normalize(Velocity, left=PLAYER_MIN_VELOCITY_arr, right=PLAYER_MAX_VELOCITY_arr)
PlayerObservation = HAGLPyMunk(PlayerObservation)

class BallObservation:
    position = Normalize(Position, left=BALL_MIN_POSITION_arr, right=BALL_MAX_POSITION_arr)
    velocity = Normalize(Velocity, left=BALL_MIN_VELOCITY_arr, right=BALL_MAX_VELOCITY_arr)
BallObservation = HAGLPyMunk(BallObservation)

TeamObservation = [PlayerObservation, T("players_count")]

class Observation:
    ball = BallObservation
    team_a = TeamObservation
    team_b = TeamObservation

class PlayerAction:
    move_direction = Velocity
    hit_direction = Velocity

Action = [PlayerAction, T("players_count")]

class HAGLFootball:
    def __init__(self, total_time=TOTAL_TIME, debug=False,
                 number_of_player=NUMBER_OF_PLAYER, team_B_model=RandomAgent, random_position=False,
                 team_reward_coeff=10, ball_reward_coeff=10, goal_reward=1000, message_dims_number=0,
                 is_out_rule_enabled=True):

        self.total_time = total_time
        self.debug = debug
        self.number_of_player = number_of_player
        self.random_position = random_position
        self.message_dims_number = message_dims_number
        self.is_out_rule_enabled = is_out_rule_enabled

        self.width = WIDTH
        self.height = HEIGHT

        self.ball_to_goal_reward_coefficient = ball_reward_coeff
        self.run_to_ball_reward_coefficient = team_reward_coeff
        self.goal_reward = goal_reward

        self.action_space = Action
        self.observation_space = Observation

        # create space
        self.space = pymunk.Space()
        self.space.gravity = 0, 0

        # Amount of simple damping to apply to the space.
        # A value of 0.9 means that each body will lose 10% of its velocity per second.
        self.space.damping = 0.95

        # create walls
        self._setup_walls(WIDTH, HEIGHT)

        # Teams
        self.team_A = Team(self.space, WIDTH, HEIGHT,
                           player_radius=PLAYER_RADIUS,
                           player_weight=PLAYER_WEIGHT,
                           player_max_velocity=PLAYER_MAX_VELOCITY,
                           color=pygame.Color("red"),  # red
                           side="left",
                           player_number=self.number_of_player,
                           message_dims_number=self.message_dims_number)

        self.team_B = Team(self.space, WIDTH, HEIGHT,
                           player_radius=PLAYER_RADIUS,
                           player_weight=PLAYER_WEIGHT,
                           player_max_velocity=PLAYER_MAX_VELOCITY,
                           color=pygame.Color("blue"),  # blue
                           side="right",
                           player_number=self.number_of_player,
                           message_dims_number=self.message_dims_number)

        self.player_arr = self.team_A.player_array + self.team_B.player_array

        # Ball
        self.ball = Ball(self.space, WIDTH * 0.5, HEIGHT * 0.5,
                         mass=BALL_WEIGHT,
                         max_velocity=BALL_MAX_VELOCITY,
                         radius=BALL_RADIUS,
                         elasticity=0.2)

        self.reset()
        self.observation, self.inverse_obs = self._get_observation()
        self.team_B_model = team_B_model(self)

    def _position_to_initial(self):

        self.team_A.set_position_to_initial(self.random_position)
        self.team_B.set_position_to_initial(self.random_position)
        self.ball.set_position(WIDTH * 0.5, HEIGHT * 0.5)

        # set the ball velocity to zero
        self.ball.body.velocity = 0, 0

        # after set position, need to step the space so that the object
        # move to the target position
        self.space.step(0.0001)

        self.observation, self.inverse_obs = self._get_observation()

    def reset(self):
        self.current_time = 0
        self.ball_owner_side = random.choice(["left", "right"])
        self._position_to_initial()
        return self._get_observation()[0], {"players_count": self.number_of_player}

    # normalized observation
    def _get_observation(self):

        get_body = lambda obj: obj.body
        team_A_body = list(map(get_body, self.team_A.player_array))
        team_B_body = list(map(get_body, self.team_B.player_array))

        observation = Observation()

        observation.ball = self.ball.body
        observation.team_a = team_A_body
        observation.team_b = team_B_body

        inverse_observation = Observation()
        inverse_observation.ball = inverse_by_x(self.ball.body, BALL_MIN_POSITION_arr[0], BALL_MAX_POSITION_arr[0])

        player_inverse = lambda player: inverse_by_x(player.copy(), PLAYER_MIN_POSITION[0], PLAYER_MAX_POSITION[0])
        inverse_observation.team_b = list(map(player_inverse, team_A_body))
        inverse_observation.team_a = list(map(player_inverse, team_B_body))

        return observation, inverse_observation

    def _setup_walls(self, width, height):
        # Create walls.
        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 0), (0, height/2-GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (0, height/2+GOAL_SIZE/2), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (0, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width, 0), (width, height/2-GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (width, height/2+GOAL_SIZE/2), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (0, 0), (width, 0), 1)
        ]

        static_goal = [
            pymunk.Segment(
                self.space.static_body,
                (-2, height/2-GOAL_SIZE/2), (-2, height/2+GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (-2, height/2-GOAL_SIZE/2), (0, height/2-GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (-2, height/2+GOAL_SIZE/2), (0, height/2+GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (width+2, height/2-GOAL_SIZE/2), (width+2, height/2+GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (width, height/2-GOAL_SIZE/2), (width+2, height/2-GOAL_SIZE/2), 1),
            pymunk.Segment(
                self.space.static_body,
                (width, height/2+GOAL_SIZE/2), (width+2, height/2+GOAL_SIZE/2), 1)
        ]

        for s in (static + static_goal):
            s.friction = 1.
            s.group = 1
            s.collision_type = 1

        self.static = static
        self.space.add(*static)
        self.static_goal = static_goal
        self.space.add(*static_goal)

    def render(self, mode):
        padding = 5
        ax = plt.axes(xlim=(0 - padding, WIDTH + padding),
                      ylim=(0 - padding, HEIGHT + padding))
        ax.set_aspect("equal")
        o = pymunk.matplotlib_util.DrawOptions(ax)
        self.space.debug_draw(o)
        plt.show()

    # return true and wall index if the ball is in contact with the walls

    def ball_contact_wall(self):
        wall_index, i = -1, 0
        for wall in self.static:
            if self.ball.shape.shapes_collide(wall).points != []:
                wall_index = i
                return True, wall_index
            i += 1
        return False, wall_index

    def check_and_fix_out_bounds(self):
        out, wall_index = self.ball_contact_wall()
        if out:
            bx, by = self.ball.get_position()
            dbx, dby, dpx, dpy = 0, 0, 0, 0

            if wall_index == 1 or wall_index == 0:  # left bound
                dbx, dpx = 3.5, 1
            elif wall_index == 3 or wall_index == 4:
                dbx, dpx = -3.5, -1
            elif wall_index == 2:
                dby, dpy = -3.5, -1
            else:
                dby, dpy = 3.5, 1

            self.ball.set_position(bx + dbx, by + dby)
            self.ball.body.velocity = 0, 0

            if self.ball_owner_side == "right":
                get_ball_player = random.choice(self.team_A.player_array)
                self.ball_owner_side = "left"
            elif self.ball_owner_side == "left":
                get_ball_player = random.choice(self.team_B.player_array)
                self.ball_owner_side = "right"
            else:
                print("invalid side")

            get_ball_player.set_position(bx + dpx, by + dpy)
            get_ball_player.body.velocity = 0, 0
        else:
            pass
        return out

    # return true if score

    def ball_contact_goal(self):
        goal = False
        for goal_wall in self.static_goal:
            goal = goal or self.ball.shape.shapes_collide(
                goal_wall).points != []
        return goal

    # if player has contact with ball and move, let the ball move with the player.

    def _ball_move_with_player(self, player):
        if self.ball.has_contact_with(player):
            self.ball.body.velocity = player.body.velocity
        else:
            pass

    def _process_action(self, player, action):
        player.apply_force_to_player(PLAYER_FORCE_LIMIT * action.move_direction.x,
                                     PLAYER_FORCE_LIMIT * action.move_direction.y)

        if self.ball.has_contact_with(player):
            self.ball.apply_force_to_ball(BALL_FORCE_LIMIT * action.hit_direction.x,
                                          BALL_FORCE_LIMIT * action.hit_direction.y)

    # action space
    # 1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
    # 2) Action Keys: Discrete 5  - noop[0], dash[1], shoot[2], press[3], pass[4] - params: min: 0, max: 4
    def step(self, team_A_action):
        team_B_action = self.team_B_model.predict(self.inverse_obs)

        for action in team_B_action:
            action.move_direction.x *= -1
            action.hit_direction.x *= -1

        team_A_init_distance_arr = self._ball_to_team_distance_arr(self.team_A)
        team_B_init_distance_arr = self._ball_to_team_distance_arr(self.team_B)

        ball_init = self.ball.get_position()

        done = False
        reward = [0, 0]

        self._process_team_action(self.team_A.player_array, team_A_action)
        self._process_team_action(self.team_B.player_array, team_B_action)

        # fix the out of bound situation
        out = self.check_and_fix_out_bounds() if self.is_out_rule_enabled else False

        # step environment using pymunk
        self.space.step(TIME_STEP)
        self.observation, self.inverse_obs = self._get_observation()

        # get reward
        if not out:
            ball_after = self.ball.get_position()

            reward[0] += self.get_team_reward(team_A_init_distance_arr, self.team_A)
            reward[1] += self.get_team_reward(team_A_init_distance_arr, self.team_A)

            reward[0] += self.get_ball_reward(ball_init, ball_after, [WIDTH, HEIGHT/2])
            reward[1] += self.get_ball_reward(ball_init, ball_after, [0, HEIGHT/2])

        if self.ball_contact_goal():
            bx, _ = self.ball.get_position()

            reward[0] += self.goal_reward if bx > WIDTH - 2 else -self.goal_reward
            reward[1] += -self.goal_reward if bx > WIDTH - 2 else self.goal_reward

            self._position_to_initial()
            self.ball_owner_side = random.choice(["left", "right"])
            # done = True

        self.current_time += TIME_STEP

        if self.current_time > self.total_time:
            done = True

        return self.observation, reward[0], done, {}, {}

    def _process_team_action(self, team_players, team_actions):
        for player, action in zip(team_players, team_actions):
            self._process_action(player, action)
            # change ball owner if any contact
            if self.ball.has_contact_with(player):
                self.ball_owner_side = player.side

    def _ball_to_team_distance_arr(self, team):
        distance_arr = []
        bx, by = self.ball.get_position()
        for player in team.player_array:
            px, py = player.get_position()
            distance_arr.append(math.sqrt((px-bx)**2 + (py-by)**2))
        return np.array(distance_arr)

    def get_team_reward(self, init_distance_arr, team):

        after_distance_arr = self._ball_to_team_distance_arr(team)
        difference_arr = init_distance_arr - after_distance_arr

        if self.number_of_player == 5:
            return np.max([difference_arr[3], difference_arr[4]]) * self.run_to_ball_reward_coefficient
        else:
            return np.max(difference_arr) * self.run_to_ball_reward_coefficient

    def get_ball_reward(self, ball_init, ball_after, goal):

        _, ball_a_to_goal = get_vec(ball_after, goal)
        _, ball_i_to_goal = get_vec(ball_init, goal)

        return (ball_i_to_goal - ball_a_to_goal) * self.ball_to_goal_reward_coefficient

    def set_team_b_model(self, model):
        self.team_B_model = model
