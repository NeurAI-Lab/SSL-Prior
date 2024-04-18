import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from util.loggers import CsvLogger
from util.utils import save_checkpoint, log
from util.test import test_all_datasets
from criterion import NTXent, KLLoss, SimSiamLoss


def get_criteria(args):
    """
    Loss criterion / criteria selection for training
    """
    criteria = {
        'simclr': [NTXent(args), args.train.loss.criterion_weight],
        'simsiam': [SimSiamLoss(), args.train.loss.criterion_weight],
        'vicreg': [(), args.train.loss.criterion_weight],
        'cog1': [(), args.train.loss.criterion_weight],
        'cog2': [(), args.train.loss.criterion_weight],
        'cog3_vic': [(), args.train.loss.criterion_weight],
        'cog3_sim': [SimSiamLoss(), args.train.loss.criterion_weight],
        'cog3_simclr': [NTXent(args), args.train.loss.criterion_weight],
        'cog3_adv_sim': [SimSiamLoss(), args.train.loss.criterion_weight],
        #                    #sim, invar, var, covar, hcr, rkd, sim

    }

    return criteria[args.train.model]


def write_scalar(writer, total_loss, loss_dict, iteration):
    """
    Add Loss scalars to tensorboard
    """
    writer.add_scalar("Total_Loss/train", total_loss , global_step=iteration)
    for loss_name, loss_item in loss_dict.items():
        writer.add_scalar("{}_Loss/train".format(loss_name), loss_item, global_step=iteration)


def train_one_epoch(args, train_loader, model, optimizer, scheduler, epoch, writer):
    """
    Train one epoch of SSL model
    """
    total_loss = 0
    loss_per_criterion = {}
    for i, (data, targets) in enumerate(train_loader):

        optimizer.zero_grad()
        loss, loss_dict = model(data)

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if i % 100 == 0:
            log("Batch {}/{}. Loss: {}.  Time elapsed: {} ".format(i, len(train_loader), loss.item(),
                                                                   datetime.now() - args.start_time))
        total_loss += loss.item()

        iteration = (epoch * len(train_loader)) + i
        write_scalar(writer, total_loss, loss_dict, iteration)

    return total_loss


def trainSSL(args, model, train_loader, optimizer, writer, scheduler=None):
    """
    Train a SSL model
    """
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        log('Model converted to DP model with {} cuda devices'.format(torch.cuda.device_count()))
    model = model.to(args.device)
    csv_logger = CsvLogger(args)

    for epoch in tqdm(range(1, args.train.epochs + 1)):
        model.train()

        total_loss = train_one_epoch(args, train_loader, model, optimizer, scheduler, epoch, writer)
        log("Epoch {}/{}. Total Loss: {}.   Time elapsed: {} ".
            format(epoch, args.train.epochs, total_loss / len(train_loader), datetime.now() - args.start_time))

        # Save checkpoint after every epoch
        # path = save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch, filename='checkpoint.pth'.format(epoch))
        # if os.path.exists:
        #     state_dict = torch.load(path, map_location=args.device)
        #     model.load_state_dict(state_dict)

        # Save the model at specific checkpoints
        if epoch > 0 and epoch % args.train.save_model == 0:
            if torch.cuda.device_count() > 1:
                save_checkpoint(state_dict=model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))
            else:
                save_checkpoint(state_dict=model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_model_{}.pth'.format(epoch))

    log("Total training time {}".format(datetime.now() - args.start_time))

    # Test the SSl Model
    if torch.cuda.device_count() > 1:
        test_all_datasets(args, writer, model.module, csv_logger)
    else:
        test_all_datasets(args, writer, model, csv_logger)

    csv_logger.write(args)

    writer.close()
