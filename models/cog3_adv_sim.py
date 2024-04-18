import torch
from torch.autograd import Variable
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


@registry.METHODS.register("cog3_adv_sim")
class cog3_adv_sim(nn.Module):
    def __init__(self, args, img_size, backbone='resnet18', criterion=None):
        super().__init__()
        self.args = args
        self.criterion = criterion
        self.f, args.projection_size = get_encoder(backbone, img_size)
        self.fsh, _ = get_encoder(backbone, img_size)
        if img_size >= 100:
            args.projection_size = self.f.fc.out_features

        # projection MLP
        self.g = self.gsh = nn.Sequential(
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
        self.h = self.hsh = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, args.train.n_proj),
        )

    def forward(self, data, targets=None, optimizer=None):

        x = data[0].cuda(device=self.args.device)
        y = data[1].cuda(device=self.args.device)
        x_sh = data[2].cuda(device=self.args.device)

        x = self.f(x)
        fx = torch.flatten(x, start_dim=1)
        zx = self.g(fx)
        px = self.h(zx)

        y = self.f(y)
        fy = torch.flatten(y, start_dim=1)
        zy = self.g(fy)
        py = self.h(zy)

        final_loss = {}
        final_loss['loss'] = 0

        loss_n1 = self.criterion[0](zx, zy, px, py)

        loss_adv, f_adv = self.madry_loss(self.fsh, data[0].cuda(device=self.args.device), targets.cuda(device=self.args.device), optimizer)

        loss_n3 = distil_Loss(self.args, fx, f_adv, self.args.train.loss.loss_mode, 0)
        loss_n3.collate(model_num='m3', final_loss=final_loss)

        final_loss['loss_nce_m1'] = loss_n1
        final_loss['loss'] += loss_n1
        final_loss['loss'] += loss_adv

        return final_loss['loss'], final_loss


    def madry_loss(self,model,
                   x1,
                   y,
                   optimizer,
                   step_size=0.007,
                   epsilon=0.031,
                   perturb_steps=10,
                   reduce=True
                   ):

        model.eval()
        # generate adversarial example
        x_adv = x1.detach() + 0.001 * torch.randn(x1.shape).cuda().detach()

        for _ in range(perturb_steps):
            x_adv.requires_grad_()

            with torch.enable_grad():
                # loss_kl = F.cross_entropy(F.log_softmax(model(x_adv), dim=1), y)

                out_adv = model(x_adv)
                out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
                out_adv = torch.flatten(out_adv, start_dim=1)
                loss = F.cross_entropy(out_adv, y)

            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x1 - epsilon), x1 + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)

        model.train()

        x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
        # zero gradient
        optimizer.zero_grad()

        # calculate robust loss
        out_adv = model(x_adv)
        out_adv = out_adv[0] if isinstance(out_adv, tuple) else out_adv
        out_adv = torch.flatten(out_adv, start_dim=1)

        if reduce:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(reduction='none')

        loss_adv = criterion(out_adv, y)

        return loss_adv, out_adv
