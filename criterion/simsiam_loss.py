"""
SimSiam: Exploring Simple Siamese Representation Learning
Code from their pseudocode
"""
import torch.nn as nn
import torch.nn.functional as F

class SimSiamLoss(nn.Module):

    def __init__(self):
        super(SimSiamLoss, self).__init__()

    def forward(self, zx, zy, px, py):
        zx = F.normalize(zx, dim=1)
        zy = F.normalize(zy, dim=1)
        px = F.normalize(px, dim=1)
        py = F.normalize(py, dim=1)

        loss = -(zx.detach() * py).sum(dim=1).mean()
        loss += -(zy.detach() * px).sum(dim=1).mean()
        return loss / 2