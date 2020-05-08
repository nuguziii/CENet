import pprint
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torch.optim as optim

from model import Network
from data_generator import LiverDataset
from loss import Loss
from utils import create_logger, save_checkpoint
from test import test

def train(opt):
    logger, final_output_dir, tb_log_dir = create_logger(opt, 'train')

    logger.info(pprint.pformat(opt))
    logger.info(opt)

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = Network(1)
    model = torch.nn.DataParallel(model, device_ids=opt.gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion = Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # Data loading code
    train_dataset = LiverDataset('train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1
    begin_epoch = 0

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, opt.lr, 0.2,
        last_epoch=last_epoch
    )

    for epoch in range(begin_epoch, opt.epoch):
        for idx, (image, label) in enumerate(train_loader):
            lr_scheduler.step()

            # train for one epoch
            output, contour_output, shape_output = model(image)
            loss = criterion(output, shape_output, contour_output, label, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate on validation set
            perf_indicator = test()

            if perf_indicator >= best_perf:
                best_perf = perf_indicator
                best_model = True
                best_model_state_file = os.path.join(
                    final_output_dir, 'best_model.pth'
                )
                logger.info('=> saving best model state to {}'.format(
                    best_model_state_file)
                )
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'epoch': epoch + 1,
                'model': opt.description,
                'state_dict': model.state_dict(),
                'best_state_dict': model.module.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
            }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()