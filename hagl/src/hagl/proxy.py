from collections import OrderedDict

from hagl import HAGLType
from hagl.base_functions import get_hagl_vars, compile_type, deconstruct
from hagl.physic_types import Vector
from hagl.template import get_template
from hagl.exceptions import ProxyException


class Proxy(HAGLType):

    def __init__(self, target_type, proxy):
        self.target_type = target_type
        self.proxy = proxy

    def gym_type(self, template_values):
        t_target_type = get_template(self.target_type, template_values)
        return compile_type(t_target_type, template_values)

    def construct(self, gym_value, template_values):
        raise ProxyException("Construction Proxy from Gymnasium values is not supported")

    def deconstruct(self, box2d_value, template_values):
        t_target_type = get_template(self.target_type, template_values)
        t_proxy = get_template(self.proxy, template_values)

        result = OrderedDict()
        type_vars = get_hagl_vars(t_target_type)
        for field_name in type_vars:
            field_value = type_vars[field_name]
            try:
                hagl_value = t_proxy[field_name](getattr(box2d_value, field_name))
            except KeyError:
                raise ProxyException("You need to use field names similar to Box2D body")
            deconstructed_value = deconstruct(field_value, hagl_value, template_values)
            result[field_name] = deconstructed_value
        return result


def proxy_vec2d(vec2d):
    vec = Vector()
    vec.array = [vec2d.x, vec2d.y]
    return vec


def proxy_asis(something):
    return something


Box2DProxy = dict(
    position=proxy_vec2d,
    angle=proxy_asis,
    angularDamping=proxy_asis,
    angularVelocity=proxy_asis,
    linearDamping=proxy_asis,
    linearVelocity=proxy_vec2d
)


class HAGLBox2D(Proxy):

    def __init__(self, target_type):
        super().__init__(target_type, Box2DProxy)


PyMunkProxy = dict(
    position=proxy_vec2d,
    velocity=proxy_vec2d
)


class HAGLPyMunk(Proxy):

    def __init__(self, target_type):
        super().__init__(target_type, PyMunkProxy)
