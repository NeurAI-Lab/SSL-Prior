import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarityPreserving(nn.Module):
    """
    Similarity-Preserving Knowledge Distillation
    Author: https://github.com/HobbitLong/RepDistiller
    https://arxiv.org/pdf/1907.09682.pdf
    """

    def __init__(self):
        super(SimilarityPreserving, self).__init__()
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, zxs, zys, zxt, zyt, temperature):
        bsz = zxs.shape[0]
        sim_s = self.similarity_f(zxs.unsqueeze(1), zys.unsqueeze(0)) / temperature
        sim_s = torch.nn.functional.normalize(sim_s)
        sim_t = self.similarity_f(zxt.unsqueeze(1), zyt.unsqueeze(0)) / temperature
        sim_t = torch.nn.functional.normalize(sim_t)
        G_diff = sim_t - sim_s
        loss = (G_diff * G_diff).view(-1, 1).sum(0) / (bsz * bsz)
        return loss.squeeze()


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

    if not squared:
        res = res.sqrt()

    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class RKdAngle(nn.Module):
    """
    Angle Loss function implemented in Relational Knowledge Distillation
    https://arxiv.org/abs/1904.05068
    """
    def forward(self, student, teacher, targets,  global_step):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        return loss


class RkdDistance(nn.Module):
    """
    Distance Loss function implemented in Relational Knowledge Distillation
    https://arxiv.org/abs/1904.05068
    """
    def forward(self, student, teacher, targets, global_step):
        with torch.no_grad():
            t_d = pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='mean')
        return loss