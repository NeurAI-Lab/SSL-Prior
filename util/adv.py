
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np

mean = [0.4914, 0.4822, 0.4465]
std = [0.247, 0.243, 0.261]


def forward_transform(image):
    image[:, 0, :, :] = (image[:, 0, :, :] - mean[0]) / std[0]
    image[:, 1, :, :] = (image[:, 1, :, :] - mean[1]) / std[1]
    image[:, 2, :, :] = (image[:, 2, :, :] - mean[2]) / std[2]
    return image

def back_transform(image, scale=255.):
    image[:, 0, :, :] = (image[:, 0, :, :] * std[0]) + mean[0]
    image[:, 1, :, :] = (image[:, 1, :, :] * std[1]) + mean[1]
    image[:, 2, :, :] = (image[:, 2, :, :] * std[2]) + mean[2]

    return image * scale

def clamp_tensor(image, upper_bound, lower_bound):
    image = torch.where(image > upper_bound, upper_bound, image)
    image = torch.where(image < lower_bound, lower_bound, image)
    return image

def get_eps_bounds(eps, x_adv, tensor_std):

    pert_epsilon = torch.ones_like(x_adv) * eps #/ tensor_std
    pert_upper = x_adv + pert_epsilon
    pert_lower = x_adv - pert_epsilon
    upper_bound = torch.ones_like(x_adv) * 255
    lower_bound = torch.zeros_like(x_adv)
    upper_bound = forward_transform(upper_bound)
    lower_bound = forward_transform(lower_bound)
    upper_bound = torch.min(upper_bound, pert_upper)
    lower_bound = torch.max(lower_bound, pert_lower)

    return upper_bound, lower_bound

def base_attack(images, targets, epsilon, device, step_size):
    # normalizing the epsilon value
    eps = epsilon #* 255.0
    # from https://github.com/columbia/MTRobust/blob/99d17939161fd7425cba5f32472ca85db0cace64/learning/attack.py
    tensor_std = images.std(dim=(0,2,3))
    tensor_std = tensor_std.unsqueeze(0)
    tensor_std = tensor_std.unsqueeze(2)
    tensor_std = tensor_std.unsqueeze(2).float()
    x_adv = images.clone()
    upper_bound, lower_bound = get_eps_bounds(eps, x_adv, tensor_std)

    x_adv = x_adv.cuda()
    upper_bound = upper_bound.to(device)
    lower_bound = lower_bound.to(device)
    tensor_std = tensor_std.to(device)
    ones_x = torch.ones_like(images).float()
    ones_x = ones_x.to(device)

    step_size_tensor = ones_x * step_size #/ tensor_std

    noise = torch.FloatTensor(images.size()).uniform_(-eps, eps)
    noise = noise.to(device)
    #noise = noise / tensor_std
    x_adv = x_adv + noise

    return x_adv, upper_bound, lower_bound, step_size_tensor
# =============================================================================
# Robustness Evaluation
# =============================================================================

def _attack_norm(model,
                  lin_model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device):

    x_adv, upper_bound, lower_bound, step_size_tensor = base_attack(X, y, epsilon, device, step_size)
    X_pgd = clamp_tensor(x_adv, upper_bound, lower_bound)
    X_pgd = Variable(X_pgd, requires_grad=True)

    out_x = model.f(X)
    feature = torch.flatten(out_x, start_dim=1)
    out = lin_model(feature)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            out_x = model.f(X_pgd)
            feature = torch.flatten(out_x, start_dim=1)
            out = lin_model(feature)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()

        eta = step_size_tensor * X_pgd.grad.data.sign()
        X_pgd = X_pgd + eta
        X_pgd = clamp_tensor(X_pgd, upper_bound, lower_bound)
        # x_adv.detach()
        X_pgd = Variable(X_pgd.data, requires_grad=True)

    out_x = model.f(X_pgd)
    feature = torch.flatten(out_x, start_dim=1)
    out = lin_model(feature)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def _pgd_whitebox(model,
                  lin_model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  step_size,
                  random,
                  device,
                  save_imgs = False,
                  ind = 0
                  ):

    out_x = model.f(X)
    feature = torch.flatten(out_x, start_dim=1)
    out = lin_model(feature)
    out = out[0] if isinstance(out, tuple) else out
    err = (out.data.max(1)[1] != y.data).float().sum()

    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()
        with torch.enable_grad():
            out_x = model.f(X_pgd)
            feature = torch.flatten(out_x, start_dim=1)
            out = lin_model(feature)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.CrossEntropyLoss()(out, y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    out_x = model.f(X_pgd)
    feature = torch.flatten(out_x, start_dim=1)
    out = lin_model(feature)
    out = out[0] if isinstance(out, tuple) else out
    err_pgd = (out.data.max(1)[1] != y.data).float().sum()
    # print('err pgd (white-box): ', err_pgd)
    return err, err_pgd


def eval_adv_robustness(
    model,
    lin_model,
    data_loader,
    epsilon,
    num_steps,
    step_size,
    random=True,
    device='cuda',
    save_imgs = False
):

    """
    evaluate model by white-box attack
    """
    model.eval()
    lin_model.eval()

    for param in model.parameters():
        param.requires_grad = False
    for param in lin_model.parameters():
        param.requires_grad = False

    robust_err_total = 0
    natural_err_total = 0

    for data, target in tqdm(data_loader, desc='robustness'):
        data, target = data.to(device), target.to(device)
        # pgd attack
        X, y = Variable(data, requires_grad=True), Variable(target)
        # err_natural, err_robust = _attack_norm(model, lin_model, X, y, epsilon, num_steps, step_size, random, device)

        err_natural, err_robust = _pgd_whitebox(model, lin_model, X, y, epsilon, num_steps, step_size, random, device, save_imgs)
        robust_err_total += err_robust
        natural_err_total += err_natural

    nat_err = natural_err_total.item()
    successful_attacks = robust_err_total.item()
    total_samples = len(data_loader.dataset)

    rob_acc = (total_samples - successful_attacks) / total_samples
    nat_acc = (total_samples - nat_err) / total_samples

    print('=' * 30)
    print("eps = {}".format(epsilon))
    print(f"Adversarial Robustness = {rob_acc * 100} % ({total_samples - successful_attacks}/{total_samples})")
    print(f"Natural Accuracy = {nat_acc * 100} % ({total_samples - nat_err}/{total_samples})")

    return nat_acc, rob_acc