import numpy as np
import pymunk
from pymunk.vec2d import Vec2d


class Player():

    def __init__(self, space, x, y, mass=70, radius=1.5, max_velocity=10,
                 elasticity=0.2, color=(1, 0, 0, 1), message_dims_number=0, side="left"):
        self.color_dims_number = message_dims_number
        self.space = space
        self.max_velocity = max_velocity
        self.color = color
        self.message = np.zeros(message_dims_number)
        self.side = side
        self.body, self.shape = self._setup_player(
            space, x, y, mass, radius, elasticity)

    def get_position(self):
        x, y = self.body.position
        return [x, y]

    def get_velocity(self):
        vx, vy = self.body.velocity
        return [vx, vy]

    def get_message(self):
        return self.message.tolist()

    def get_observation(self):
        if self.color_dims_number:
            return self.get_position() + self.get_velocity() + self.get_message()
        else:
            return self.get_position() + self.get_velocity()

    def set_position(self, x, y):
        self.body.position = x, y

    def set_message(self, message):
        self.message = message

    # apply force in the center of the player
    # fx: force in x direction
    # fy: force in y direction

    def apply_force_to_player(self, fx, fy):
        # self.body.apply_force_at_local_point((fx,fy), point=(0, 0))
        self.body.apply_impulse_at_local_point((fx, fy), point=(0, 0))

    def _setup_player(self, space, x, y, mass, radius, elasticity):
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, moment)

        body.position = x, y
        body.start_position = Vec2d(x, y)

        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            l = body.velocity.length
            if l > self.max_velocity:
                scale = self.max_velocity / l
                body.velocity = body.velocity * scale

        body.velocity_func = limit_velocity

        shape = pymunk.Circle(body, radius)
        shape.color = self.color
        shape.elasticity = elasticity
        self.space.add(body, shape)
        return body, shape
