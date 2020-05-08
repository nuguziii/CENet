
# 1. 3Dircadb: https://www.ircad.fr/research/3d-ircadb-01/ (20)
# 2. zenodo: https://zenodo.org/record/1169361#.XrPDA2gzZjU (label)
#         - TCIA: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT (43)
#         - BTCV (Abdomen): https://www.synapse.org/#!Synapse:syn3193805 (47)

import random
import glob
import os
import pydicom
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom

from dataset import get_data_filelist, read_dicom, read_nii

root_dir = 'E:\INFINITT'

class LiverDataset(Dataset):
    def __init__(self, state='train'):
        self.state = state
        self.im_to_label, self.image_list = get_data_filelist(state)

    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.im_to_label[image_file]
        image = self._load_image(image_file)
        label = self._load_image(label_file)
        liver_label = self._get_liver_label(label)

        # pre-processing
        seed = random.random() >= 0.8
        mode = random.uniform(0, 3)
        image = self._transform(self._resize(self._normalize(self._windowing(image))), seed, mode)
        liver_label = self._transform(self._resize(self._normalize(self._windowing(liver_label))), seed, mode)
        return image, liver_label

    def __len__(self):
        return len(self.image_list)

    def _load_image(self, file):
        if os.path.isdir(file):
            return read_dicom(file)
        else:
            return read_nii(file)

    def _get_liver_label(self, label):
        return (label * (label==6)) / 6

    def _windowing(self, image): # window width = 700
        return np.clip(image, -340, 360)

    def _normalize(self, image):
        inp = (image.astype(np.float32) / 700.)
        mean = np.mean(inp)
        std = np.std(inp)
        return (inp - mean) / std

    def _resize(self, image):
        c, w, h = image.shape
        factor = (64/c, 128/w, 128/h)
        image = zoom(image, factor)
        return image[:64, :128, :128]

    def _transform(self, image, seed, mode):
        # flip
        if seed:
            if mode==0:
                return np.flip(image)
            elif mode==1:
                return np.rot90(image)
            elif mode==2:
                return np.rot90(image, k=2)
            elif mode==3:
                return np.rot90(image, k=3)
        else:
            return image

    def _cutout(self):
        pass

if __name__ == '__main__':
    data = LiverDataset()
    image, label = data.__getitem__(1)