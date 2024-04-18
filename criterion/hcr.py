import torch
import torch.nn as nn
import torch.nn.functional as F


class HCR(nn.Module):
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def pairwise_dist(self, x):
        x_square = x.pow(2).sum(dim=1)
        prod = x @ x.t()
        pdist = (x_square.unsqueeze(1) + x_square.unsqueeze(0) - 2 * prod).clamp(min=self.eps)
        pdist[range(len(x)), range(len(x))] = 0.
        return pdist

    def pairwise_prob(self, pdist):
        return torch.exp(-pdist)

    def hcr_loss(self, h, g):
        q1, q2 = self.pairwise_prob(self.pairwise_dist(h)), self.pairwise_prob(self.pairwise_dist(g))
        return -1 * (q1 * torch.log(q2 + self.eps)).mean() + -1 * ((1 - q1) * torch.log((1 - q2) + self.eps)).mean()

    def forward(self, logits, projections):
        loss = self.hcr_loss(F.normalize(logits, dim=1), F.normalize(projections, dim=1).detach())

        return loss