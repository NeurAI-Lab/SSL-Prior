from util import registry
from .helper import LinearEvaluation

from .simclr import SimCLR
from .simsiam import SimSiam
from .vicreg import VICReg
from .cog1 import cog1
from .cog2 import cog2
from .cog3_vic import cog3_vic
from .cog3_sim import cog3_sim
from .cog3_simclr import cog3_simclr
# from .cog3_adv_sim import cog3_adv_sim

__all__ = [
    "build_model",
    "SimCLR",
    "SimSiam",
    "VICReg",
    "cog1",
    "cog2",
    "cog3_vic",
    "cog3_sim",
    "cog3_simclr",
    # "cog3_adv_sim"
]

def build_model(args, img_size, backbone='resnet18', criterion=None):
    name = args.train.model
    return registry.METHODS[name](args, img_size, backbone, criterion)
