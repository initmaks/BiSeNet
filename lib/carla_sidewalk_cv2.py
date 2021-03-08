#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json
import glob

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
    {"name":"Unlabeled",    "id":0,       "color":[0, 0, 0],          "trainId":255},
    {"name":"Building",     "id":1,       "color":[70, 70, 70],       "trainId":2},
    {"name":"Fence",        "id":2,       "color":[100, 40, 40],      "trainId":2}, 
    {"name":"Other",        "id":3,       "color":[55, 90, 80],       "trainId":3},
    {"name":"Pedestrian",   "id":4,       "color":[220, 20, 60],      "trainId":5},
    {"name":"Pole",         "id":5,       "color":[153, 153, 153],    "trainId":3},
    {"name":"RoadLine",     "id":6,       "color":[157, 234, 50],     "trainId":3},
    {"name":"Road",         "id":7,       "color":[128, 64, 128],     "trainId":1},   # SWAPPED! 0-1
    {"name":"SideWalk",     "id":8,       "color":[244, 35, 232],     "trainId":0},   # SWAPPED! 1-0
    {"name":"Vegetation",   "id":9,       "color":[107, 142, 35],     "trainId":3},
    {"name":"Vehicles",     "id":10,      "color":[0, 0, 142],        "trainId":6},
    {"name":"Wall",         "id":11,      "color":[102, 102, 156],    "trainId":2},
    {"name":"TrafficSign",  "id":12,      "color":[220, 220, 0],      "trainId":3},
    {"name":"Sky",          "id":13,      "color":[70, 130, 180],     "trainId":4},
    {"name":"Ground",       "id":14,      "color":[81, 0, 81],        "trainId":1}, # not sampling sidewalk points on the "ground"
    {"name":"Bridge",       "id":15,      "color":[150, 100, 100],    "trainId":2},
    {"name":"RailTrack",    "id":16,      "color":[230, 150, 140],    "trainId":3},
    {"name":"GuardRail",    "id":17,      "color":[180, 165, 180],    "trainId":2},
    {"name":"TrafficLight", "id":18,      "color":[250, 170, 30],     "trainId":7},
    {"name":"Static",       "id":19,      "color":[110, 190, 160],    "trainId":3},
    {"name":"Dynamic",      "id":20,      "color":[170, 120, 50],     "trainId":3},
    {"name":"Water",        "id":21,      "color":[45, 60, 150],      "trainId":1}, # unclear - just marking as road_after / non-walkable
    {"name":"Terrain",      "id":22,      "color":[145, 170, 100],    "trainId":1}, # TERRAIN IS ALWAYS ROAD_AFTER
]


class CarlaScapes(Dataset):
    '''
    '''
    def __init__(self, pattern, trans_func=None):
        super(Dataset, self).__init__()
        self.trans_func = trans_func

        self.img_paths = glob.glob(pattern + "_rgb.png")
        self.lb_paths = glob.glob(pattern + "_label.np.npy")
        assert len(self.img_paths) == len(self.lb_paths)

        self.n_cats = 8
        self.lb_ignore = 255
        self.lb_map = np.arange(256).astype(np.uint8)
        for el in labels_info:
            self.lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor( # use city scapes mean/var
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

    def __getitem__(self, idx):
        impth, lbpth = self.img_paths[idx], self.lb_paths[idx]
        img = cv2.imread(impth)[:, :, ::-1]
        label = np.load(lbpth,allow_pickle=True)
        if not self.lb_map is None:
            label = self.lb_map[label]
        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return len(self.img_paths)

def get_data_loader(pattern, ims_per_gpu, scales, cropsize, max_iter=None, mode='train', distributed=True):
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

    ds = CarlaScapes(pattern, trans_func=trans_func)

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