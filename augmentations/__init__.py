from util import registry
from .test_transform import Test_transform

from .simclr_transform import SimCLRTransform
from .simsiam_transform import SimSiamTransform
from .vicreg_transform import VICRegTransform
from .cog1_transform import cog1Transform
from .cog2_transform import cog2Transform
from .cog3_transform import cog3Transform

__all__ = [
    "build_transform",
    "SimCLRTransform",
    "SimSiamTransform",
    "VICRegTransform",
    "cog1Transform",
    "cog2_transform",
    "cog3_transform",
    "Test_transform",
]

def build_transform(args):
    if 'cog3' in args.train.model:
        name = "cog3_transform"
    else:
        name = args.train.model + '_transform'
    return registry.TRANSFORMS[name](args.train.dataset)
