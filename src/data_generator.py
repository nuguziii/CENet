
# 1. 3Dircadb: https://www.ircad.fr/research/3d-ircadb-01/ (20)
# 2. zenodo: https://zenodo.org/record/1169361#.XrPDA2gzZjU (label)
#         - TCIA: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT (43)
#         - BTCV (Abdomen): https://www.synapse.org/#!Synapse:syn3193805 (47)

import random
import os
import numpy as np
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from scipy.ndimage.interpolation import rotate
from skimage import feature
import cv2

from dataset import get_data_filelist, read_dicom, read_nii_8, read_nii_16
from utils import save_img_to_nib

class LiverDataset(Dataset):
    def __init__(self, state='train', root_dir='E:\INFINITT\dataset', imShow=False):
        self.state = state
        self.im_to_label, self.image_list = get_data_filelist(state, root_dir)
        self.imShow = imShow

    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.im_to_label[image_file]
        image_original = read_nii_16(image_file)
        label_original = read_nii_8(label_file)

        liver_label = self._get_liver_label(label_original)

        # pre-processing
        if self.state == 'test':
            seed = False
        else:
            seed = random.random() <= 0.8
        mode = 1 #int(random.uniform(0, 4))

        image = self._clip(self._transform(self._resize(self._normalize(self._windowing(image_original)), order=3), seed, mode))
        liver_label = self._clip(self._transform(self._resize(liver_label, order=0), seed, mode))

        # contour ground truth
        liver_contour = np.zeros_like(liver_label)
        for i in range(liver_contour.shape[-1]):
            liver_contour[:,:,i] = feature.canny(liver_label[:,:,i]*255)

        # shape ground truth (smaller version)
        liver_shape = self._clip(zoom(liver_label, 0.5))

        if self.imShow:
            save_img_to_nib(image, './', 'test_img_'+str(index))
            save_img_to_nib(liver_label, './', 'test_label_'+str(index))
            save_img_to_nib(liver_contour, './', 'test_contour_' + str(index))
            save_img_to_nib(liver_shape, './', 'test_shape_' + str(index))

        # expand_dims and (c,w,h)
        image = np.expand_dims(np.transpose(image, (2, 0, 1)), axis=0)
        liver_label = np.transpose(liver_label, (2, 0, 1))
        liver_shape = np.transpose(liver_shape, (2, 0, 1))
        liver_contour = np.transpose(liver_contour, (2, 0, 1))

        return image, liver_label, liver_contour, liver_shape

    def __len__(self):
        return len(self.image_list)

    def _get_liver_label(self, label):
        return np.ones_like(label) * (label==6)

    def _windowing(self, image): # window width = 700
        return np.clip(image, -340, 360)

    def _normalize(self, image):
        return ((image.astype(np.float32) + 340) / 700.)
        #mean = np.mean(inp)
        #std = np.std(inp)
        #return (inp - mean) / std

    def _resize(self, image, order=0):
        w, h, c = image.shape
        factor = (128/w, 128/h, 64/c)
        image = zoom(image, factor, order=order)
        return image[:128, :128, :64]

    def _eq_hist(self, image):
        eq = cv2.equalizeHist(image)
        for i in range(image.shape[2]):
            image[:,:,i] = cv2.equalizeHist(image[:,:,i].astype(np.uint16))
        return image

    def _transform(self, image, seed, mode):
        # flip
        if seed:
            if mode==0:
                return rotate(image, angle=30, reshape=False)
            elif mode==1:
                return rotate(image, angle=-30, reshape=False)
            elif mode==2:
                return rotate(image, angle=10, reshape=False)
            elif mode==3:
                return rotate(image, angle=-10, reshape=False)

        return image

    def _cutout(self):
        pass

    def _clip(self, image):
        return np.clip(image, 0, 1)

if __name__ == '__main__':
    data = LiverDataset(imShow=True)

    for i in range(1):
        print(i)
        image, liver_label, liver_contour, liver_shape = data.__getitem__(i)
        #print(image.shape, liver_label.shape, liver_contour.shape, liver_shape.shape, np.histogram(liver_label))