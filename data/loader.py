import os
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, STL10, ImageFolder
from util.utils import tiny_imagenet, log

def trainloaderSSL(args, transform, imagenet_split='train'):
    """
    Load training data through DataLoader
    """
    if args.train.dataset.name == 'CIFAR100':
        train_dataset = CIFAR100(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'ImageNet':
        train_dataset = ImageFolder(os.path.join(args.train.dataset.data_dir, imagenet_split), transform=transform)
    elif args.train.dataset.name == 'CIFAR10':
        train_dataset = CIFAR10(args.train.dataset.data_dir, train=True, download=True, transform=transform)
    elif args.train.dataset.name == 'TinyImageNet':
        train_dataset = tiny_imagenet(args.train.dataset.data_dir, train=True, transform=transform)
    elif args.train.dataset.name == 'STL10':
        train_dataset = STL10(args.train.dataset.data_dir, split="unlabeled", download=True, transform=transform)
    elif args.train.dataset.name == 'ImageNet100':
        train_dataset = ImageFolder(os.path.join(args.train.dataset.data_dir, imagenet_split), transform=transform)
    elif args.train.dataset.name == 'Imagenette':
        train_dataset = ImageFolder(os.path.join(args.train.dataset.data_dir, imagenet_split), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train.batchsize, shuffle=True, drop_last=True, num_workers=args.train.num_workers)
    log("Took {} time to load data!".format(datetime.now() - args.start_time))

    return train_loader
