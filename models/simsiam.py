import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper import get_encoder
from util import registry

@registry.METHODS.register("simsiam")
class SimSiam(nn.Module):
    """
    Exploring Simple Siamese Representation Learning
    https://arxiv.org/abs/2011.10566
    """
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super(SimSiam, self).__init__()

        self.args = args
        self.f, args.projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f.fc.out_features

        # projection MLP
        self.g = nn.Sequential(
            nn.Linear(args.projection_size, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            # nn.Linear(2048, 2048),
            # nn.BatchNorm1d(2048),
            # nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
        )
        # predictor MLP
        self.h = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.train.n_proj),
        )

        self.criterion = nn.CosineSimilarity(dim=1).cuda(args.device)

    def forward(self, data):

        x1 = data[0].cuda(device=self.args.device)
        x2 = data[1].cuda(device=self.args.device)

        f1 = self.f(x1)
        f1 = torch.flatten(f1, start_dim=1)
        z1 = self.g(f1)
        p1 = self.h(z1)

        f2 = self.f(x2)
        f2 = torch.flatten(f2, start_dim=1)
        z2 = self.g(f2)
        p2 = self.h(z2)

        # zx = F.normalize(zx, dim=1)
        # zy = F.normalize(zy, dim=1)
        # px = F.normalize(px, dim=1)
        # py = F.normalize(py, dim=1)

        # Multiple loss aggregation
        loss_dict= {}
        # loss_dict['nce'] = self.criterion[0](zx, zy, px, py)
        loss_nce = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) * 0.5
        loss_dict['nce'] = loss_nce

        return loss_dict['nce'], loss_dict
