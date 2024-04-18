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


@registry.METHODS.register("cog3_simclr")
class cog3_simclr(nn.Module):
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super().__init__()
        self.args = args
        self.criterion=criterion
        self.f, args.projection_size = get_encoder(backbone, img_size)
        self.fsh, _ = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f.fc.out_features

        self.g = self.gsh = nn.Sequential(
                                nn.Linear(args.projection_size, 512, bias=False),
                                nn.BatchNorm1d(512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, args.train.n_proj, bias=True)
                               )

    def forward(self, data):
        x = data[0].cuda(device=self.args.device)
        y = data[1].cuda(device=self.args.device)
        x_sh = data[2].cuda(device=self.args.device)

        x = self.f(x)
        feat_x = torch.flatten(x, start_dim=1)
        zx = self.g(feat_x)
        y = self.f(y)
        feat_y = torch.flatten(y, start_dim=1)
        zy = self.g(feat_y)

        x_sh = self.fsh(x_sh)
        fsh = torch.flatten(x_sh, start_dim=1)
        zsh = self.gsh(fsh)

        final_loss = {}
        final_loss['loss'] = 0

        loss_n1 = self.criterion[0](zx, zy)
        loss_n3 = distil_Loss(self.args, zx, zsh, self.args.train.loss.loss_mode)

        loss_n3.collate(model_num='m3', final_loss=final_loss)

        final_loss['loss_m1'] = loss_n1
        final_loss['loss'] += loss_n1

        return final_loss['loss'], final_loss

