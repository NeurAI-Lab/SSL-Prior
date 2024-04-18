import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper import get_encoder
from util.model_util import off_diagonal, FullGatherLayer
from util import registry

def Projector(args, embedding, mlp):
    mlp_spec = f"{embedding}-{mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)


@registry.METHODS.register("cog1")
class cog1(nn.Module):
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super().__init__()
        self.args = args
        self.criterion = criterion

        mlp = "2048-2048-2048"
        self.num_features = int(mlp.split("-")[-1])
        self.f, args.projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f1.fc.out_features
        self.g = Projector(args, args.projection_size, mlp)
        self.f2, args.projection_size = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f2.fc.out_features
        self.g2 = Projector(args, args.projection_size, mlp)

    def forward(self, x, y):
        x = self.f(x)
        fx = torch.flatten(x, start_dim=1)
        zx = self.g(fx)

        y = self.f2(y)
        fy = torch.flatten(y, start_dim=1)
        zy = self.g2(fy)

        repr_loss = F.mse_loss(zx, zy)
        # x = torch.cat(FullGatherLayer.apply(x), dim=0)
        # y = torch.cat(FullGatherLayer.apply(y), dim=0)
        zx = zx - zx.mean(dim=0)
        zy = zy - zy.mean(dim=0)

        std_x = torch.sqrt(zx.var(dim=0) + 0.0001)
        std_y = torch.sqrt(zy.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (zx.T @ zx) / (self.args.train.batchsize - 1)
        cov_y = (zy.T @ zy) / (self.args.train.batchsize - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss_dict= {}
        loss_dict['var'] = repr_loss
        loss_dict['invar'] = std_loss
        loss_dict['cov'] = cov_loss

        loss = (
            self.criterion[1] * repr_loss
            + self.criterion[2] * std_loss
            + self.criterion[3] * cov_loss
        )
        return loss, loss_dict

