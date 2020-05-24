
# 1. 3Dircadb: https://www.ircad.fr/research/3d-ircadb-01/ (20)
# 2. zenodo: https://zenodo.org/record/1169361#.XrPDA2gzZjU (label)
#         - TCIA: https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT (43)
#         - BTCV (Abdomen): https://www.synapse.org/#!Synapse:syn3193805 (47)

import os, glob
import pydicom
import nibabel as nib
import numpy as np

def read_dicom(dir):
    file_list = glob.glob(dir + '/*.dcm')
    volume_list = []
    for filename in file_list:
        volume_list.append(pydicom.dcmread(filename).pixel_array)
    return np.array(volume_list)

def read_nii_8(filename):
    return np.array(nib.load(filename).dataobj, dtype=np.uint8)

def read_nii_16(filename):
    return np.array(nib.load(filename).dataobj, dtype=np.int16)

def get_data_filelist(state, root_dir):
    '''
    :return: data dictionary {original image path : label image path}
    '''
    data = {}
    filelist = []
    filetemp = glob.glob(os.path.join(root_dir, state) + '/*')

    for file in filetemp:
        num = file.split('.')[0].split('\\')[-1]
        label_file = glob.glob(os.path.join(root_dir, 'label') + '/mask_' + str(num) + '.*')[0]
        data[file] = label_file
        filelist.append(file)

    '''
    #3Dircadb
    dircadb_dir_list = glob.glob(os.path.join(root_dir, '3Dircadb') + '/*')
    for dircadb_dir in dircadb_dir_list:
        imfile = os.path.join(dircadb_dir, 'PATIENT_DICOM')
        data[imfile] = os.path.join(dircadb_dir, 'MASKS_DICOM/liver')
        filelist.append(imfile)
    '''
    return data, filelist

if __name__ == '__main__':
    #print(read_nii('E:\INFINITT\zenodo\\tcia\label0002.nii.gz').shape)
    #print(read_dicom('E:\INFINITT\TCIA\Pancreas-CT\PANCREAS_0001\\11-24-2015-PANCREAS0001-Pancreas-18957\Pancreas-99667').shape)
    get_data_filelist('test', 'E:\INFINITT\dataset')