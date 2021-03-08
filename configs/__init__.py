from .bisenetv2_custom import cfg as bisenetv2_custom_cfg

class cfg_dict(object):

    def __init__(self, d):
        self.__dict__ = d


cfg_factory = dict(
    bisenetv2_custom=cfg_dict(bisenetv2_custom_cfg),
)
