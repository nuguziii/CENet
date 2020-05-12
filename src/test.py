import pprint
import os, time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import numpy as np

from model import Network
from data_generator import LiverDataset
from utils import create_logger, save_checkpoint, AverageMeter, save_img_to_nib, pred_image
from evaluate import dc, hd, assd, sensitivity, precision

def test(opt):
    logger, final_output_dir, result_dir = create_logger(opt, 'test')

    logger.info(pprint.pformat(opt))
    logger.info(opt)

    model = Network(in_channels=1)
    model = torch.nn.DataParallel(model, device_ids=[opt.gpus]).cuda()

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
    model.load_state_dict(torch.load(model_file), strict=False)

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    dc_val = AverageMeter()
    hd_val = AverageMeter()
    assd_val = AverageMeter()
    s_val = AverageMeter()
    p_val = AverageMeter()

    end = time.time()
    for idx, (image, label, contour_label, shape_label) in enumerate(test_loader):
        data_time.update(time.time() - end)

        # test model
        output, _, _ = model(image)

        output = np.transpose(np.squeeze(output.detach().cpu().numpy(), axis=0), (0,2,3,1))
        output = pred_image(output).astype(np.float)

        label = np.transpose(np.squeeze(label.detach().cpu().numpy(), axis=0), (1,2,0)).astype(np.float)

        cur_dc_val = dc(output, label)
        cur_hd_val = hd(output, label)
        cur_assd_val = assd(output, label)
        cur_s_val = sensitivity(output, label)
        cur_p_val = precision(output, label)

        dc_val.update(cur_dc_val, 1)
        hd_val.update(cur_hd_val, 1)
        assd_val.update(cur_assd_val, 1)
        s_val.update(cur_s_val, 1)
        p_val.update(cur_p_val, 1)

        # save result
        save_img_to_nib(output, result_dir, 'img'+str(idx+1))

        batch_time.update(time.time() - end)
        end = time.time()

        msg = '[{0}/{1}]\t' \
              'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
              'Speed {speed:.1f} samples/s\t' \
              'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
              'DC {dc:.2f}\t\t' \
              'HD {hd:.2f}\t' \
              'ASSD {assd:.2f}\t' \
              'Sensitivity {s:.2f}\t' \
              'Precision {p:.2f}\t'.format(
                idx, len(test_loader), batch_time=batch_time,
                speed=image.size(0) / batch_time.val,
                data_time=data_time, dc=cur_dc_val, hd=cur_hd_val, assd=cur_assd_val, s=cur_s_val, p=cur_p_val)
        logger.info(msg)

    msg = '[total]\t' \
        'DC {dc.avg:.2f}\t\t'\
        'HD {hd.avg:.2f}\t'\
        'ASSD {assd.avg:.2f}\t'\
        'Sensitivity {s.avg:.2f}\t'\
        'Precision {p.avg:.2f}'.format(
        dc=dc_val, hd=hd_val, assd=assd_val, s=s_val, p=p_val)
    logger.info(msg)