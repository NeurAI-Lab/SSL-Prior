import torch
import torch.nn as nn
import torch.optim as optim

import os
from util.utils import log, save_checkpoint, tiny_imagenet
from models import LinearEvaluation
from optimizers.lars import LARC
from torch.optim.lr_scheduler import CosineAnnealingLR
from util.utils import testloader, train_or_val, eval_celeb_model
from augmentations import Test_transform


def testSSL(args, writer, model, csv_logger):
    for param in model.parameters():
        param.requires_grad = False

    linear_model = LinearEvaluation(args.projection_size, args.eval.dataset.classes)
    if torch.cuda.device_count() > 1:
        linear_model = nn.DataParallel(linear_model)
    linear_model = linear_model.to(args.device)

    scheduler = None
    if args.eval.optimizer.name == 'adam':
        optimizer = optim.Adam(linear_model.parameters(), lr=args.eval.optimizer.lr, weight_decay=args.eval.optimizer.weight_decay)
    elif args.eval.optimizer.name == 'SGD':
        optimizer = optim.SGD(linear_model.parameters(), lr=args.eval.optimizer.lr,
                               weight_decay=args.eval.optimizer.weight_decay, momentum=args.eval.optimizer.momentum)

    if args.eval.optimizer.scheduler:
            scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=3e-4)

    transform = Test_transform(args.eval.dataset.img_size, args.eval.dataset)
    train_loader, val_loader, test_loader = testloader(args, args.eval.dataset.name, transform,
                                                             args.eval.batchsize, args.eval.dataset.data_dir)

    loss_criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    log('Testing SSL Model on {}.................'.format(args.eval.dataset.name))
    _, ck_name = os.path.split(args.eval.model_path)
    # ck_name = args.eval.linear_name
    for epoch in range(1, args.eval.epochs + 1):
        loss_epoch, accuracy_epoch = train_or_val(args, train_loader, model, linear_model, loss_criterion, optimizer, scheduler, train=True)
        log(f"Epoch [{epoch}/{args.eval.epochs}]\t Loss: {loss_epoch / len(train_loader)}\t Accuracy: {accuracy_epoch / len(train_loader)}")

        loss_epoch1, accuracy_epoch1 = train_or_val(args, val_loader, model, linear_model, loss_criterion, train=False)
        val_accuracy = accuracy_epoch1 / len(val_loader)
        log(f"Epoch [{epoch}/{args.eval.epochs}] \t Validation accuracy {val_accuracy}")
        if best_acc < val_accuracy:
            best_acc = val_accuracy
            log('Best accuracy achieved so far: {}'.format(best_acc))
            if torch.cuda.device_count() > 1:
                # Save DDP model's module
                save_checkpoint(state_dict=linear_model.module.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name))
            else:
                save_checkpoint(state_dict=linear_model.state_dict(), args=args, epoch=epoch,
                                filename='checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name))
        writer.add_scalar("Accuracy/train{}".format(args.eval.dataset.name), accuracy_epoch / len(train_loader), epoch)
        writer.add_scalar("Accuracy/val{}".format(args.eval.dataset.name), accuracy_epoch1 / len(val_loader), epoch)

    # Load best linear model and run inference on test set
    state_dict = torch.load(os.path.join(args.model_dir, 'checkpoint_best_linear_model_{}_{}.pth'.format(args.eval.dataset.name, ck_name)), map_location=args.device)
    linear_best_model = LinearEvaluation(args.projection_size, args.eval.dataset.classes)
    linear_best_model.load_state_dict(state_dict)
    linear_best_model = linear_best_model.cuda()
    if args.eval.dataset.name == 'CelebA':
        overall, blond_male, nonblonde_male, blond_female, nonblond_female = eval_celeb_model(model, linear_best_model, test_loader)
        test_acc = overall
        csv_logger.log(overall)
        csv_logger.log_celeb(blond_male, nonblonde_male, blond_female, nonblond_female)
    else:
        test_loss, test_acc = train_or_val(args, test_loader, model, linear_best_model, loss_criterion, train=False)
    test_acc = test_acc / len(test_loader)
    log(f" Test accuracy : {test_acc}")
    csv_logger.log(test_acc)
    writer.add_text("Test Accuracy {} :".format(args.eval.dataset.name), "{}".format(test_acc))

    csv_logger.write(args)

def test_all_datasets(args, writer, model, csv_logger):
    """
    Test all datasets for linear evaluation
    """

    testSSL(args, writer, model, csv_logger)
