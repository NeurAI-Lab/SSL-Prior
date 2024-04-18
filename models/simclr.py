import torch.nn as nn
import torch
import torch.nn.functional as F
from models.helper import get_encoder
from util import registry

@registry.METHODS.register("simclr")
class SimCLR(nn.Module):
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super(SimCLR, self).__init__()

        self.args = args
        self.criterion = criterion
        self.f, args.projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f.fc.out_features

        self.g = nn.Sequential(
                                nn.Linear(args.projection_size, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, args.train.n_proj, bias=True)
                               )

    def forward(self, data):
        x = data[0].cuda(device=self.args.device)
        y = data[1].cuda(device=self.args.device)

        x = self.f(x)
        feat_x = torch.flatten(x, start_dim=1)
        zx = self.g(feat_x)

        y = self.f(y)
        feat_y = torch.flatten(y, start_dim=1)
        zy = self.g(feat_y)

        loss_dict= {}
        loss_dict['simclr'] = self.criterion[0](zx, zy)

        return loss_dict['simclr'], loss_dict
