from hagl.base_types import HAGLType

DIMENSIONS_TEMPLATE_NAME = "dimensions"

class Template(HAGLType):

    def __init__(self, template_name):
        self.template_name = template_name

    def name(self):
        return self.template_name

    def gym_type(self, template_values):
        return template_values[self.template_name]

def get_template(template_value, template_dict):
    if isinstance(template_value, Template):
        return template_dict[template_value.name()]
    return template_value

T = Template