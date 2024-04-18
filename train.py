import os
import torch
import random
from datetime import datetime
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
# sys.path.insert(0, '.')
main_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from util.utils import logger, summary_writer, log
from util.train_util import trainSSL, get_criteria
from augmentations import build_transform
from config.option import Options
from models import build_model
from util.distributed import init_distributed_mode
from data.loader import trainloaderSSL


if __name__ == "__main__":
    args = Options().parse()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    init_distributed_mode(args)

    args.writer = summary_writer(args)
    logger(args)
    args.start_time = datetime.now()
    log("Starting at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))

    criterion = get_criteria(args)
    model = build_model(args, args.train.dataset.img_size, backbone=args.train.backbone, criterion=criterion)
    transform = build_transform(args)

    train_loader = trainloaderSSL(args, transform)
    scheduler = None
    if args.train.optimizer.name == 'adam':
        optimizer = Adam(model.parameters(), lr=args.train.optimizer.lr, weight_decay=args.train.optimizer.weight_decay)
    elif args.train.optimizer.name ==  'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.train.optimizer.lr, momentum=args.train.optimizer.momentum, weight_decay=args.train.optimizer.weight_decay)

    if args.train.optimizer.scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-4)

    trainSSL(args, model, train_loader, optimizer, args.writer, scheduler)



