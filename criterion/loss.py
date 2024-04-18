import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from criterion.hcr import HCR
from criterion.simsiam_loss import SimSiamLoss
from util.model_util import off_diagonal
from criterion import NTXent

class distil_Loss():
    def __init__(self, args, n1, n2, loss_mode, num_features=0, p1=None, p2=None):
        self.args = args
        self.loss_mode = loss_mode
        self.num_features = num_features
        self.n1 = n1
        self.n2 = n2
        self.p1 = p1
        self.p2 = p2
        self.loss_dist = {}
        self.hcr = HCR(args.train.loss.eps)
        self.nce = SimSiamLoss()
        self.temperature = 1.0
        self.loss_dist['loss'] = 0

        self.get_loss()

    def get_loss(self):

        if 'kl' in self.loss_mode:
            self.loss_dist['loss_kl'] = 0
            loss = self.kl_loss(self.n1, self.n2, self.temperature)
            self.loss_dist['loss_kl'] = loss

        if 'fitnet' in self.loss_mode:
            self.loss_dist['loss_fitnet'] = 0
            n = random.randint(3, 4)
            loss = self.l2_loss(self.st_feat[n], self.tch_feat[n])
            self.loss_dist['loss_fitnet'] = loss

        if 'invar' in self.loss_mode:
            self.loss_dist['loss_invar'] = 0
            loss = self.l2_loss(self.n1, self.n2)
            self.loss_dist['loss_invar'] = loss

        if 'varn' in self.loss_mode:
            self.loss_dist['loss_varn'] = 0
            loss = self.var_loss(self.n1, self.n2)
            self.loss_dist['loss_varn'] = loss

        if 'covar' in self.loss_mode:
            self.loss_dist['loss_covar'] = 0
            loss = self.covar_loss(self.n1, self.n2)
            self.loss_dist['loss_covar'] = loss

        if 'hcr' in self.loss_mode:
            self.loss_dist['loss_hcr'] = 0
            loss = self.hcr_loss(self.n1, self.n2)
            self.loss_dist['loss_hcr'] = loss

        if 'nce' in self.loss_mode:
            self.loss_dist['loss_nce'] = 0
            loss = self.nce(self.n1, self.n2, self.p1, self.p2)
            self.loss_dist['loss_nce'] = loss

        if 'cont' in self.loss_mode:
            self.loss_dist['loss_cont'] = 0
            loss = self.simclr_loss(self.n1, self.n2)
            self.loss_dist['loss_cont'] = loss

        if 'cos' in self.loss_mode:
            criterion_cos = nn.CosineSimilarity(dim=1).cuda(self.args.device)

            def pred_loss(criterion, p, z):
                return (criterion(p, z.detach()).mean())

            self.loss_dist['loss_cos'] = 0
            loss = pred_loss(criterion_cos, self.p1, self.n2)
            loss += pred_loss(criterion_cos, self.p2, self.n1)
            loss = - (loss/2)
            self.loss_dist['loss_cos'] = loss

    def kl_loss(self,n1, n2, T):

        p = F.log_softmax(n1 / T, dim=1)
        q = F.softmax(n2 / T, dim=1)
        l_kl = F.kl_div(p, q, size_average=False) * (T**2) / n1.shape[0]
        return l_kl

    def l2_loss(self, n1, n2):
        return  F.mse_loss(n1, n2)

    def var_loss(self, n1, n2):
        n1 = n1 - n1.mean(dim=0)
        n2 = n2 - n2.mean(dim=0)
        std_x = torch.sqrt(n1.var(dim=0) + 0.0001)
        std_y = torch.sqrt(n2.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2
        return std_loss

    def covar_loss(self, n1, n2):
        cov_x = (n1.T @ n1) / (self.args.train.batchsize - 1)
        cov_y = (n2.T @ n2) / (self.args.train.batchsize - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(self.num_features) +\
                   off_diagonal(cov_y).pow_(2).sum().div(self.num_features)
        return cov_loss

    def hcr_loss(self, n1, n2):
        loss_hcr = self.hcr(F.softmax(n1), n2)
        return loss_hcr

    def simclr_loss(self, n1, n2):
        crit = NTXent(self.args)

        loss_simclr = crit(n1,n2)
        return loss_simclr

    def similarity_preserving_loss(A_t, A_s):
        """Given the activations for a batch of input from the teacher and student
        network, calculate the similarity preserving knowledge distillation loss from the
        paper Similarity-Preserving Knowledge Distillation (https://arxiv.org/abs/1907.09682)
        equation 4

        Note: A_t and A_s must have the same batch size

        Parameters:
            A_t (4D tensor): activation maps from the teacher network of shape b x c1 x h1 x w1
            A_s (4D tensor): activation maps from the student network of shape b x c2 x h2 x w2

        Returns:
            l_sp (1D tensor): similarity preserving loss value
    """

        # reshape the activations
        b1, c1, h1, w1 = A_t.shape
        b2, c2, h2, w2 = A_s.shape
        assert b1 == b2, 'Dim0 (batch size) of the activation maps must be compatible'

        Q_t = A_t.reshape([b1, c1 * h1 * w1])
        Q_s = A_s.reshape([b2, c2 * h2 * w2])

        # evaluate normalized similarity matrices (eq 3)
        G_t = torch.mm(Q_t, Q_t.t())
        # G_t = G_t / G_t.norm(p=2)
        G_t = torch.nn.functional.normalize(G_t)

        G_s = torch.mm(Q_s, Q_s.t())
        # G_s = G_s / G_s.norm(p=2)
        G_s = torch.nn.functional.normalize(G_s)

        # calculate the similarity preserving loss (eq 4)
        l_sp = (G_t - G_s).pow(2).mean()

        return l_sp


    def collate(self, model_num=None, final_loss={}):
        loss_wt = self.args.train.loss.criterion_weight.__dict__

        if model_num == 'm1' or model_num == None:
            i = 0
        else:
            i = 1

        if 'invar' in self.loss_mode:
            loss_term = 'loss_invar_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_invar']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['invar'][i])
        if 'varn' in self.loss_mode:
            loss_term = 'loss_varn_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_varn']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['varn'][i])
        if 'covar' in self.loss_mode:
            loss_term = 'loss_covar_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_covar']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['covar'][i])
        if 'hcr' in self.loss_mode:
            loss_term = 'loss_hcr_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_hcr']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['hcr'][i])
        if 'nce' in self.loss_mode:
            loss_term = 'loss_nce_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_nce']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['nce'][i])
        if 'cos' in self.loss_mode:
            loss_term = 'loss_cos_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_cos']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['cos'][i])
        if 'cont' in self.loss_mode:
            loss_term = 'loss_cont_{}'.format(model_num)
            final_loss[loss_term] = self.loss_dist['loss_cont']
            final_loss['loss'] += (final_loss[loss_term] * loss_wt['cont'][i])