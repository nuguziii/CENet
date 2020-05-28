import os
import logging
import time
import torch
from pathlib import Path
import nibabel as nib
import numpy as np

def create_logger(opt, phase='train'):
    root_output_dir = Path(opt.output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    final_output_dir = root_output_dir / opt.description

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(opt.description, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(opt.log_dir) / opt.description

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    if phase is 'test':
        result_dir = final_output_dir / 'test'
        result_dir.mkdir(parents=True, exist_ok=True)
        return logger, str(final_output_dir), str(result_dir)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def save_checkpoint(states, is_best, output_dir, model, epoch,
                    filename='checkpoint'):
    if (epoch > 100) and (epoch % 30 == 0):
        torch.save(model, os.path.join(output_dir, filename + '_' + str(epoch) + '.pth'))
    else:
        torch.save(model, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['best_state_dict'],
                   os.path.join(output_dir, 'model_best_state_dict.pth'))
        torch.save(model, os.path.join(output_dir, 'model_best.pth'))

def save_img_to_nib(img, path, file_name):
    # img = (w, h, d)
    img = nib.Nifti1Image(img, np.eye(4))
    nib.save(img, os.path.join(path, file_name+'.nii.gz'))

def pred_image(img):
    # img = (c, w, h, d)
    return np.argmax(img, axis=0)

def visualize_compare(pred, label): # (128, 128, 64)
    pred = 0.7 * pred
    label = 0.3 * label
    return pred + label

def visualize(image, pred):
    return image * pred



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0