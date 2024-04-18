import torch
import torch.nn as nn
import torch.nn.functional as F
from models.helper import get_encoder
from util import registry
from criterion.loss import distil_Loss

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


@registry.METHODS.register("cog2")
class cog2(nn.Module):
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super().__init__()
        self.args = args

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

        loss_n1 = distil_Loss(self.args, zx, zy)
        loss_n2 = distil_Loss(self.args, zx, zy)

        final_loss = {}
        final_loss['loss'] = 0

        loss_n1.collate(model_num='m1', final_loss=final_loss)
        loss_n2.collate(model_num='m2', final_loss=final_loss)

        return final_loss['loss'], final_loss

