#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json

import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import cv2
import numpy as np

import lib.transform_cv2 as T
from lib.sampler import RepeatedDistSampler
from lib.base_dataset import BaseDataset, TransformationTrain, TransformationVal

# "road_under"    : 0 
# "road_after"    : 1 
# "large_static"  : 2 # buildings, walls, fences
# "obstacles"     : 3
# "sky"           : 4
# "people"        : 5
# "vehicles"      : 6
# "traffic_light" : 7

labels_info = [
    {"name": "unlabeled",               "id": 0, "color": [0, 0, 0],        "trainId": 255},
    {"name": "ego vehicle",             "id": 1, "color": [0, 0, 0],        "trainId": 255},
    {"name": "rectification border",    "id": 2, "color": [0, 0, 0],        "trainId": 255},
    {"name": "out of roi",              "id": 3, "color": [0, 0, 0],        "trainId": 255},
    {"name": "static",                  "id": 4, "color": [0, 0, 0],        "trainId": 3},
    {"name": "dynamic",                 "id": 5, "color": [111, 74, 0],     "trainId": 3},
    {"name": "ground",                  "id": 6, "color": [81, 0, 81],      "trainId": 0},      # ground is shared by cars (==road)
    {"name": "road",                    "id": 7, "color": [128, 64, 128],   "trainId": 0},      # ROAD -> SIDEWALK FLIP
    {"name": "sidewalk",                "id": 8, "color": [244, 35, 232],   "trainId": 1},      # ROAD -> SIDEWALK FLIP
    {"name": "parking",                 "id": 9, "color": [250, 170, 160],  "trainId": 0},    
    {"name": "rail track",              "id": 10, "color": [230, 150, 140], "trainId": 3},      
    {"name": "building",                "id": 11, "color": [70, 70, 70],    "trainId": 2},      # 
    {"name": "wall",                    "id": 12, "color": [102, 102, 156], "trainId": 2},      #
    {"name": "fence",                   "id": 13, "color": [190, 153, 153], "trainId": 2},      #
    {"name": "guard rail",              "id": 14, "color": [180, 165, 180], "trainId": 2},      #
    {"name": "bridge",                  "id": 15, "color": [150, 100, 100], "trainId": 2},      #
    {"name": "tunnel",                  "id": 16, "color": [150, 120, 90],  "trainId": 2},      #
    {"name": "pole",                    "id": 17, "color": [153, 153, 153], "trainId": 3},      #
    {"name": "polegroup",               "id": 18, "color": [153, 153, 153], "trainId": 3},      #
    {"name": "traffic light",           "id": 19, "color": [250, 170, 30],  "trainId": 7},      #
    {"name": "traffic sign",            "id": 20, "color": [220, 220, 0],   "trainId": 3},      #
    {"name": "vegetation",              "id": 21, "color": [107, 142, 35],  "trainId": 3},      #
    {"name": "terrain",                 "id": 22, "color": [152, 251, 152], "trainId": 1},      # TERRAIN IS ALWAYS ROAD_AFTER
    {"name": "sky",                     "id": 23, "color": [70, 130, 180],  "trainId": 4},      #
    {"name": "person",                  "id": 24, "color": [220, 20, 60],   "trainId": 5},      #
    {"name": "rider",                   "id": 25, "color": [255, 0, 0],     "trainId": 5},      #
    {"name": "car",                     "id": 26, "color": [0, 0, 142],     "trainId": 6},      #
    {"name": "truck",                   "id": 27, "color": [0, 0, 70],      "trainId": 6},      #
    {"name": "bus",                     "id": 28, "color": [0, 60, 100],    "trainId": 6},      #
    {"name": "caravan",                 "id": 29, "color": [0, 0, 90],      "trainId": 6},      #
    {"name": "trailer",                 "id": 30, "color": [0, 0, 110],     "trainId": 6},      #
    {"name": "train",                   "id": 31, "color": [0, 80, 100],    "trainId": 6},      #
    {"name": "motorcycle",              "id": 32, "color": [0, 0, 230],     "trainId": 6},      #
    {"name": "bicycle",                 "id": 33, "color": [119, 11, 32],   "trainId": 6},      #
    {"name": "license plate",           "id": -1, "color": [0, 0, 142],     "trainId": 255}
]

class CityScapesMini(BaseDataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None, mode='train'):
        super(CityScapesMini, self).__init__(
                dataroot, annpath, trans_func, mode)
        self.n_cats = 8
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )


def get_data_loader(datapth, annpath, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=True):
    if mode == 'train':
        trans_func = TransformationTrain(scales, cropsize)
        batchsize = ims_per_gpu
        shuffle = True
        drop_last = True
    elif mode == 'val':
        trans_func = TransformationVal()
        batchsize = ims_per_gpu
        shuffle = False
        drop_last = False

    ds = CityScapes(datapth, annpath, trans_func=trans_func, mode=mode)

    if distributed:
        assert dist.is_available(), "dist should be initialzed"
        if mode == 'train':
            assert not max_iter is None
            n_train_imgs = ims_per_gpu * dist.get_world_size() * max_iter
            sampler = RepeatedDistSampler(ds, n_train_imgs, shuffle=shuffle)
        else:
            sampler = torch.utils.data.distributed.DistributedSampler(
                ds, shuffle=shuffle)
        batchsampler = torch.utils.data.sampler.BatchSampler(
            sampler, batchsize, drop_last=drop_last
        )
        dl = DataLoader(
            ds,
            batch_sampler=batchsampler,
            num_workers=4,
            pin_memory=True,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batchsize,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=4,
            pin_memory=True,
        )
    return dl
