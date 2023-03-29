from collections import OrderedDict

import gymnasium.spaces

from hagl import HAGLType
from hagl.base_functions import get_hagl_vars, compile_type, construct, deconstruct
from hagl.physic_types import Vector
from hagl.template import get_template
from hagl.exceptions import ProxyException

def proxy_box2dvec(box2dvec):
    vec = Vector()
    vec.x = box2dvec.x
    vec.y = box2dvec.y
    vec.array = [box2dvec.x, box2dvec.y]
    return vec

def proxy_box2dfloat(box2dfloat):
    return box2dfloat

Box2DProxy = dict(
    position=proxy_box2dvec,
    angle=proxy_box2dfloat,
    angularDamping=proxy_box2dfloat,
    angularVelocity=proxy_box2dfloat,
    linearDamping=proxy_box2dfloat,
    linearVelocity=proxy_box2dvec
)

class Proxy(HAGLType):

    def __init__(self, target_type, proxy):
        self.target_type = target_type
        self.proxy = proxy

    def gym_type(self, template_values):
        t_target_type = get_template(self.target_type, template_values)
        return compile_type(t_target_type, template_values)

    def construct(self, gym_value, template_values):
        raise ProxyException("Construction Box2D from Gymnasium values is not supported")

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

class HAGLBox2D(Proxy):

    def __init__(self, target_type):
        super().__init__(target_type, Box2DProxy)