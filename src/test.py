import pprint
import os, time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from data_generator import LiverDataset
from utils import create_logger, save_checkpoint, AverageMeter, save_img_to_nib, pred_image, visualize_compare, visualize
from evaluate import dc, hd95, assd, sensitivity, precision

def test(opt):
    logger, final_output_dir, result_dir = create_logger(opt, 'test')

    logger.info(pprint.pformat(opt))
    logger.info(opt)

    # Data loading code
    test_dataset = LiverDataset('test', opt.data_dir)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model_file = os.path.join(
        final_output_dir, opt.model_name
    )

    logger.info("=> loading model '{}'".format(model_file))
    model = torch.load(model_file)
    #model = torch.nn.DataParallel(model, device_ids=[opt.gpus]).cuda()
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    dc_val = AverageMeter()
    hd_val = AverageMeter()
    assd_val = AverageMeter()
    s_val = AverageMeter()
    p_val = AverageMeter()

    end = time.time()
    for idx, (image, label) in enumerate(test_loader):
        data_time.update(time.time() - end)

        # test model
        output = model(image)

        img = np.transpose(image.detach().cpu().numpy()[0,0], (1, 2, 0)).astype(np.float)
        pred = np.transpose(output.detach().cpu().numpy()[0], (0, 2, 3, 1))
        pred = pred_image(pred).astype(np.float)
        lab = np.transpose(label.detach().cpu().numpy()[0], (1, 2, 0)).astype(np.float)

        dc_val.update(dc(pred, lab), 1)
        hd_val.update(hd95(pred, lab), 1)
        assd_val.update(assd(pred, lab), 1)
        s_val.update(sensitivity(pred, lab), 1)
        p_val.update(precision(pred, lab), 1)

        # save result
        save_img_to_nib(pred, result_dir, 'img' + str(idx + 1))
        save_img_to_nib(lab, result_dir, 'lab' + str(idx + 1))

        # visualize
        vis_cmp = visualize_compare(pred, lab)
        save_img_to_nib(vis_cmp, result_dir, 'vis_cmp' + str(idx + 1))

        vis = visualize(img, pred)
        save_img_to_nib(vis, result_dir, 'masked_img' + str(idx + 1))


        batch_time.update(time.time() - end)
        end = time.time()

        msg = '[{0}/{1}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'DC {dc.val:.2f}\t\t' \
              'HD {hd.val:.2f}\t' \
              'ASSD {assd.val:.2f}\t' \
              'Sensitivity {s.val:.2f}\t' \
              'Precision {p.val:.2f}\t'.format(
                idx, len(test_loader), batch_time=batch_time,
                speed=image.size(0) / batch_time.val,
                data_time=data_time, dc=dc_val, hd=hd_val, assd=assd_val, s=s_val, p=p_val)
        logger.info(msg)

    msg = '[total]\t' \
        'DC {dc.avg:.2f}\t\t'\
        'HD {hd.avg:.2f}\t'\
        'ASSD {assd.avg:.2f}\t'\
        'Sensitivity {s.avg:.2f}\t'\
        'Precision {p.avg:.2f}'.format(
        dc=dc_val, hd=hd_val, assd=assd_val, s=s_val, p=p_val)
    logger.info(msg)