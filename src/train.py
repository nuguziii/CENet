import pprint
import os, time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
import torch.optim as optim
import numpy as np

from model import Network
from data_generator import LiverDataset
from loss import Loss
from utils import create_logger, save_checkpoint, AverageMeter, save_img_to_nib, pred_image
from evaluate import dc, hd, assd, sensitivity, precision

def validate(output, label):
    pred = np.transpose(output.detach().cpu().numpy()[0], (0, 2, 3, 1))
    pred = pred_image(pred).astype(np.float)
    lab = np.transpose(label.detach().cpu().numpy()[0], (1, 2, 0)).astype(np.float)
    return dc(pred, lab)

def train(opt):
    logger, final_output_dir, tb_log_dir = create_logger(opt, 'train')

    logger.info(pprint.pformat(opt))
    logger.info(opt)

    writer = SummaryWriter(log_dir=tb_log_dir)

    model = Network(in_channels=1)
    model = torch.nn.DataParallel(model, device_ids=[opt.gpus]).cuda()

    # define loss function (criterion) and optimizer
    criterion = Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0.1)

    # Data loading code
    train_dataset = LiverDataset('train', opt.data_dir)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    best_perf = 0.0
    last_epoch = -1
    begin_epoch = 0

    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if os.path.exists(checkpoint_file) and opt.auto_resume:
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=last_epoch)
    p = 1

    model.train()
    for epoch in range(begin_epoch, opt.epoch):

        if (epoch>100 and epoch%10==0):
            p = max(p*0.9, 0.5)

        losses = AverageMeter()
        dsc = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        end = time.time()
        for idx, (image, label, contour_label, shape_label) in enumerate(train_loader):
            data_time.update(time.time() - end)
            lr_scheduler.step()

            image = image.type(torch.cuda.FloatTensor)
            label = label.type(torch.cuda.LongTensor)
            contour_label = contour_label.type(torch.cuda.LongTensor)
            shape_label = shape_label.type(torch.cuda.LongTensor)
            class_weights = torch.Tensor([0.6, 1.0]).type(torch.cuda.FloatTensor)

            # train for one epoch
            output, contour_output, shape_output = model(image)

            contour_label_tilde = contour_label * (output[:,1] < p).type(torch.cuda.LongTensor)
            output_loss, shape_loss, contour_loss = criterion(output, shape_output, contour_output, label, contour_label_tilde, shape_label, class_weights)
            loss = output_loss + shape_loss + contour_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update
            dsc.update(validate(output, label), 1)
            losses.update(loss.item(), image.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} (out:{out:.3f}/c:{c:.3f}/s:{s:.3f})\t' \
                  'DSC {dsc.val: .2f}'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    speed=image.size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses, out=output_loss.item(), c=contour_loss.item(), s=shape_loss.item(),
                    dsc=dsc)
            logger.info(msg)

        pref_indicator = dsc.avg
        if pref_indicator > best_perf:
            best_perf = pref_indicator
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
            'perf': pref_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

        writer.add_scalar('loss', losses.avg, epoch+1)
        writer.add_scalar('DSC', dsc.avg, epoch+1)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer.close()