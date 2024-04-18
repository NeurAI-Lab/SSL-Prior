import torch
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter
import logging
from util.torchlist import ImageFilelist
from augmentations import Test_transform
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10, CIFAR100, STL10
from data.dataset import *

def tiny_imagenet(data_root, img_size=64, train=True, transform=None):
    """
    TinyImageNet dataset
    """
    train_kv = "train_kv_list.txt"
    test_kv = "val_kv_list.txt"
    if train:
        train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv), transform=transform)
        return train_dataset
    else:
        train_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, train_kv), transform=transform)
        test_dataset = ImageFilelist(root=data_root, flist=os.path.join(data_root, test_kv), transform=transform)
        return train_dataset, test_dataset


def positive_mask(batch_size):
    """
    Create a mask for masking positive samples
    :param batch_size:
    :return: A mask that can segregate 2(N-1) negative samples from a batch of N samples
    """
    N = 2 * batch_size
    mask = torch.ones((N, N), dtype=torch.bool)
    mask[torch.eye(N).byte()] = 0
    for i in range(batch_size):
        mask[i, batch_size + i] = 0
        mask[batch_size + i, i] = 0
    return mask


def summary_writer(args, log_dir=None, filename_suffix=''):
    """
    Create a tensorboard SummaryWriter
    """
    base_dir = os.path.join(args.train.save_dir, args.exp)
    os.makedirs(base_dir, exist_ok=True)
    if log_dir is None:
        args.log_dir = os.path.join(base_dir) #, 'logs', datetime.now().strftime('%Y%m%d_%H%M'))
        mkdir(args.log_dir)
    else:
        args.log_dir = log_dir
    writer = SummaryWriter(log_dir=args.log_dir)
    print("logdir = {}".format(args.log_dir))
    return writer


def mkdir(path):
    """
    Creates new directory if not exists
    @param path:  folder path
    """
    if not os.path.exists(path):
        print("creating {}".format(path))
        os.makedirs(path, exist_ok=True)


def logger(args, filename=None):
    """
    Creates a basic config of logging
    @param args: Namespace instance with parsed arguments
    @param filename: None by default
    """
    if filename is None:
        filename = os.path.join(args.log_dir, 'train.log')
    else:
        filename = os.path.join(args.log_dir, filename)
    logging.basicConfig(filename=filename, level=logging.DEBUG, format='%(message)s')
    print("logfile created")


def log(msg):
    """
    print and log console messages
    @param msg: string message
    """
    print(msg)
    logging.debug(msg)


def save_checkpoint(state_dict, args, epoch, filename=None):
    """
    @param state_dict: model state dictionary
    @param args: system arguments
    @param epoch: epoch
    @param filename: filename for saving the checkpoint. Do not include whole path as path is appended in the code
    """
    args.model_dir = os.path.join(args.log_dir, "models")
    os.makedirs(args.model_dir, exist_ok=True)
    if filename is None:
        path = os.path.join(args.model_dir, "checkpoint_{}.pth".format(epoch))
    else:
        path = os.path.join(args.model_dir, filename)

    torch.save(state_dict, path)
    log("checkpoint saved at {} after {} epochs".format(path, epoch))
    return path

def object2dict(obj):
    if not  hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        element = []
        if isinstance(val, list):
            for item in val:
                element.append(object2dict(item))
        else:
            element = object2dict(val)
        result[key] = element
    return result

def get_domain_net(args):
    """ DomainNet datasets - QuickDraw, Sketch, ClipArt """
    train_kv = args.eval.dataset.name + "_train.txt"
    test_kv = args.eval.dataset.name + "_test.txt"
    train_dataset = ImageFilelist(root=args.eval.dataset.data_dir,
                                  flist=os.path.join(args.eval.dataset.data_dir, train_kv), transform=TestTransform(args.eval.dataset.img_size))
    test_dataset = ImageFilelist(root=args.eval.dataset.data_dir,
                                 flist=os.path.join(args.eval.dataset.data_dir, test_kv), transform=TestTransform(args.eval.dataset.img_size))
    return train_dataset, test_dataset


def testloader(args, dataset, transform, batchsize, data_dir, val_split=0.15):
    """
    Load test datasets
    """
    if dataset == 'CIFAR100':
        train_d = CIFAR100(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR100(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CIFAR10':
        train_d = CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_d = CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'TinyImageNet':
        train_d, test_d = tiny_imagenet(data_dir, transform=transform, train=False)
    elif dataset == 'STL10':
        train_d = STL10(data_dir, split='train', download=True, transform=transform)
        test_d = STL10(data_dir, split='test', download=True, transform=transform)
    elif dataset == 'STLTint':
        ds = STLTint()
        train_d = ds.get_dataset(data_dir, split="val_train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="val_test", download=True, transform=transform)
    elif dataset == 'quickdraw' or dataset == 'sketch' or dataset == 'clipart':
        train_d, test_d = get_domain_net(args)
    elif dataset == 'CIFAR10_imbalance':
        train_d = CIFAR10ImbalancedNoisy(root=data_dir, train=True, download=True, transform=transform, perc=args.eval.dataset.perc)
        test_d =  CIFAR10(data_dir, train=False, download=True, transform=transform)
    elif dataset == 'CelebA':
        ds = CelebA()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'Imagenet_R':
        ds = Imagenet_R()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'Imagenet_Blurry':
        ds = Imagenet_Blurry()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'Imagenet_A':
        ds = Imagenet_A()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'Imagenet_C':
        ds = Imagenet_C()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'ImageNet100':
        ds = Imagenet100()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)
    elif dataset == 'Imagenette':
        ds = Imagenette()
        train_d = ds.get_dataset(data_dir, split="train", download=True, transform=transform)
        test_d = ds.get_dataset(data_dir, split="test", download=True, transform=transform)

    # train - validation split
    val_size = int(val_split * len(train_d))
    train_size = len(train_d) - val_size
    train_d, val_d = random_split(train_d, [train_size, val_size])

    train_loader = DataLoader(train_d, batch_size=batchsize, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_d, batch_size=batchsize, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_d, batch_size=batchsize, shuffle=False, drop_last=True)
    # log("Took {} time to load data!".format(datetime.now() - args.start_time))
    return train_loader, val_loader, test_loader


def train_or_val(args, loader, method, model, criterion=None, optimizer=None, scheduler=None, train=False, ood={}):
    """
    Train Linear model
    """
    loss_epoch = 0
    accuracy_epoch = 0
    method.eval()
    if train:
        model.train()
    else:
        model.eval()
        model.zero_grad()

    for step, (x, y) in enumerate(loader):
        x = x.to(args.device)
        y = y.to(args.device)

        x = method.f(x)
        feature = torch.flatten(x, start_dim=1)
        output = model(feature)

        predicted = output.argmax(1)

        if ood:
            y_ood = [ood[n.item()] for n in y]
            y_ood = torch.tensor(y_ood).to(args.device)
            acc = (predicted == y_ood).sum().item() / y.size(0)
        else:
            acc = (predicted == y).sum().item() / y.size(0)

        accuracy_epoch += acc
        if train:
            loss = criterion(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_epoch += loss.item()
    return loss_epoch, accuracy_epoch


def eval_celeb_model(method, model, data_loader, train=False):
    y = []
    y_pred = []
    is_blond = []

    method.eval()
    if train:
        model.train()
    else:
        model.eval()
        model.zero_grad()

    for data, label, blond in data_loader:
        data, label, blond = data.cuda(), label.cuda(), blond.cuda()

        data = method.f(data)
        feature = torch.flatten(data, start_dim=1)
        scores = model(feature)

        scores = scores[0] if isinstance(scores, tuple) else scores
        _, predicted = scores.max(1)

        y.append(label.cpu())
        y_pred.append(predicted.detach().cpu())
        is_blond.append(blond.cpu())

    y = torch.cat(y).numpy()
    y_pred = torch.cat(y_pred).numpy()
    is_blond = torch.cat(is_blond).numpy()

    np.mean(y == y_pred)

    # Men Blonde
    num_samples = len(y)
    blonde_male = [(y[i] == 1) and is_blond[i] for i in range(num_samples)]
    blonde_female = [(y[i] == 0) and is_blond[i] for i in range(num_samples)]
    non_blonde_male = [(y[i] == 1) and not is_blond[i] for i in range(num_samples)]
    non_blonde_female = [(y[i] == 0) and not is_blond[i] for i in range(num_samples)]

    print('Overall:',  np.mean(y == y_pred))
    print('Blonde Male:',  np.mean(y[blonde_male] == y_pred[blonde_male]))
    print('Non Blonde Male:',  np.mean(y[non_blonde_male] == y_pred[non_blonde_male]))
    print('Blonde Female:',  np.mean(y[blonde_female] == y_pred[blonde_female]))
    print('Non Blonde Female:',  np.mean(y[non_blonde_female] == y_pred[non_blonde_female]))

    overall = np.mean(y == y_pred)
    blond_male = np.mean(y[blonde_male] == y_pred[blonde_male])
    nonblonde_male = np.mean(y[non_blonde_male] == y_pred[non_blonde_male])
    blond_female = np.mean(y[blonde_female] == y_pred[blonde_female])
    nonblond_female = np.mean(y[non_blonde_female] == y_pred[non_blonde_female])

    return overall, blond_male, nonblonde_male, blond_female, nonblond_female
