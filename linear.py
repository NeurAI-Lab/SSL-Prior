import sys
sys.path.insert(0, '.')
from models import *
from util.test import test_all_datasets
import numpy as np
from datetime import datetime
import torch
import os
from config.option import Options
from util.utils import summary_writer, logger
from util.utils import log
import logging
from util.loggers import CsvLogger
from augmentations import Test_transform

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(10)
torch.manual_seed(10)


if __name__ == '__main__':
    args = Options().parse()

    # percent_label_lst = [0.05, 0.10, 0.25, 0.5, 0.75]  # , 0.8, 1.0]
    # for perc in percent_label_lst:
    #     print(perc)
    log_dir = args.log_dir #os.path.dirname(os.path.abspath(args.eval.model_path))
    _, checkpoint = os.path.split(args.eval.model_path)
    writer = summary_writer(args, log_dir, checkpoint + '_Evaluation')
    logger(args, checkpoint + '{}_test_linear.log'.format(args.eval.dataset.name))

    # logger(args, checkpoint + '{}_test_linear_{}.log'.format(args.eval.dataset.name, perc))
    args.start_time = datetime.now()
    log("Starting testing of SSL model at  {}".format(datetime.now()))
    log("arguments parsed: {}".format(args))
    csv_logger = CsvLogger(args)
    # args.eval.dataset.perc = perc

    if args.eval.model == 'simclr':
        model = SimCLR(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'simsiam':
        model = SimSiam(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'vicreg':
        model = VICReg(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'cog3_vic':
        model = cog3_vic(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'cog3_sim':
        model = cog3_sim(args, args.eval.dataset.img_size, backbone=args.eval.backbone)
    elif args.eval.model == 'cog3_simclr':
        model = cog3_simclr(args, args.eval.dataset.img_size, backbone=args.eval.backbone)

    state_dict = torch.load(args.eval.model_path, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.cuda()
    test_all_datasets(args, writer, model, csv_logger)
    writer.close()
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)