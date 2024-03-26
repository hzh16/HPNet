import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
import torch
import numpy as np
from torch.utils import data
import h5py
import random
from collections import Counter
from util import extract_local_norm, read_las_file, plotpointcloud, extract_all_local_norm
from scipy.spatial import KDTree

EPS = np.finfo(np.float32).eps


class WHUTLSDataset(data.Dataset):
    def __init__(self, root, filename, opt = None, skip = 1, fold = 1, subsample = 10000):
        '''

        Parameters:
            root: str
                root directory of WHU-TLS Dataset
            filename: str
                file name of all poitns in WHU-TLS Dataset
            opt:
                option
            skip: int(default 1)
                skip length of the data
            subsample: int
                the number of points after subsampling
                <=0 representing not subsampling

        '''

        self.root = root
        data_path = open(os.path.join(root, filename), 'r')
        self.opt = opt
        '''self.augment_routines = [
            rotate_perturbation_point_cloud, jitter_point_cloud,
            shift_point_cloud, random_scale_point_cloud, rotate_point_cloud
        ]'''

        if 'train' in filename:
            self.augment = self.opt.augment
            self.if_normal_noise = self.opt.if_normal_noise
        else:
            self.augment = 0
            self.if_normal_noise = 0

        self.data_list = [item.strip() for item in data_path.readlines()]
        self.skip = skip

        self.subsample = subsample

        self.data_list = self.data_list[::self.skip]
        self.len = len(self.data_list)

    def __getitem__(self, index):

        ret_dict = {}
        index = index % self.len

        data_file = os.path.join(self.root, self.data_list[index])

        points = read_las_file(data_file)

        if(self.subsample > 0):
            subsampleidx = np.random.choice(len(points), self.subsample, replace = False)
            points = points[subsampleidx]

        ret_dict['gt_pc'] = points
        ret_dict['gt_normal'] = extract_all_local_norm(points)
        #ret_dict['T_gt'] = primitives.astype(int)
        #ret_dict['T_param'] = primitive_param

        return ret_dict

    def __len__(self):
        return self.len


if __name__ == '__main__':

    abc_dataset = WHUTLSDataset(
        root=
        './data/WHU-TLS',
        filename='WHUTLSdatalist.txt')

    for idx in range(len(abc_dataset)):
        example = abc_dataset[idx]
        #import ipdb
        #ipdb.set_trace()
        print(example)
        plotpointcloud(example['gt_pc'])
