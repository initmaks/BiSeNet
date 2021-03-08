#!/usr/bin/python
# -*- encoding: utf-8 -*-

import os
import os.path as osp
import json
import copy

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

cs_labels_info = [
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

gta_labels_info = [
    {"name":"unlabeled",        "id":0,     "trainId":3}, # id = sum(rgb)
    {"name":"ambiguous",        "id":185,   "trainId":3},
    {"name":"sky",              "id":380,   "trainId":4},
    {"name":"railtrack",        "id":520,   "trainId":3},
    {"name":"terrain",          "id":555,   "trainId":1},
    {"name":"tree",             "id":304,   "trainId":3},
    {"name":"vegetation",       "id":212,   "trainId":3},
    {"name":"building",         "id":210,   "trainId":2},
    {"name":"infrastructure",   "id":459,   "trainId":3},
    {"name":"fence",            "id":496,   "trainId":2},
    {"name":"billboard",        "id":190,   "trainId":3},
    {"name":"trafficlight",     "id":450,   "trainId":7},
    {"name":"trafficsign",      "id":440,   "trainId":3},
    {"name":"mobilebarrier",    "id":460,   "trainId":3},
    {"name":"firehydrant",      "id":479,   "trainId":3},
    {"name":"chair",            "id":474,   "trainId":3},
    {"name":"trash",            "id":102,   "trainId":3},
    {"name":"trashcan",         "id":163,   "trainId":3}, # sum(RGB)+1
    {"name":"person",           "id":300,   "trainId":5},
    {"name":"animal",           "id":255,   "trainId":3},
    {"name":"bicycle",          "id":162,   "trainId":6},
    {"name":"motorcycle",       "id":230,   "trainId":6},
    {"name":"car",              "id":142,   "trainId":6},
    {"name":"van",              "id":180,   "trainId":6},
    {"name":"bus",              "id":160,   "trainId":6},
    {"name":"truck",            "id":70,    "trainId":6},
    {"name":"trailer",          "id":90,    "trainId":6},
    {"name":"train",            "id":180,   "trainId":6},
    {"name":"plane",            "id":200,   "trainId":6},
    {"name":"boat",             "id":140,   "trainId":6},
]

gta_swlk_labels_info = copy.copy(gta_labels_info)
gta_swlk_labels_info.append({"name":"sidewalk", "id":511,   "trainId":0})
gta_swlk_labels_info.append({"name":"road",     "id":320,   "trainId":1})

gta_road_labels_info = copy.copy(gta_labels_info)
gta_road_labels_info.append({"name":"road",     "id":320,   "trainId":0})
gta_road_labels_info.append({"name":"sidewalk", "id":511,   "trainId":1})

class CityScapesMini(Dataset):
    '''
    '''
    def __init__(self, dataroot, annpath, trans_func=None):
        super(CityScapesMini, self).__init__()
        self.trans_func = trans_func
        self.n_cats = 8
        self.lb_ignore = 255

        self.cs_lb_map = np.arange(256).astype(np.uint8)
        for el in cs_labels_info:
            self.cs_lb_map[el['id']] = el['trainId']
        
        self.gta_sw_lb_map = np.arange(256*3).astype(np.uint8)
        for el in gta_swlk_labels_info:
            self.gta_sw_lb_map[el['id']] = el['trainId']
        
        self.gta_rd_lb_map = np.arange(256*3).astype(np.uint8)
        for el in gta_road_labels_info:
            self.gta_rd_lb_map[el['id']] = el['trainId']

        self.to_tensor = T.ToTensor(
            mean=(0.3257, 0.3690, 0.3223), # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )

        with open(annpath, 'r') as fr:
            pairs = fr.read().splitlines()
        self.img_paths, self.lb_paths, self.img_types = [], [], []
        for pair_info in pairs:
            pair_info = pair_info.split(',')
            if len(pair_info)==2:
                set_type = "cs_road"
            else:
                set_type = "gta_"
                set_type += pair_info[2]
            imgpth, lbpth = pair_info[0],pair_info[1]
            self.img_paths.append(osp.join(dataroot, imgpth))
            self.lb_paths.append(osp.join(dataroot, lbpth))
            self.img_types.append(set_type)

        assert len(self.img_paths) == len(self.lb_paths)

    def __getitem__(self, idx):
        impth, lbpth, itype = self.img_paths[idx], self.lb_paths[idx], self.img_types[idx]
        img = cv2.imread(impth)[:, :, ::-1]
        if itype == "cs_road":
            label = cv2.imread(lbpth, 0)
            label = self.cs_lb_map[label]
        else: # gta
            img = cv2.resize(img,(1024,512),interpolation=cv2.INTER_NEAREST)
            label = cv2.imread(lbpth)
            label = cv2.resize(label,(1024,512),interpolation=cv2.INTER_NEAREST)
            ixs,iys,_ = np.where(label==[32,11,119])
            label[ixs,iys]=[32,11,119] # make each use label unique
            label = label.sum(axis=2)
            if itype == "gta_road":
                label = self.gta_rd_lb_map[label]
            elif itype == "gta_sidewalk":
                label = self.gta_sw_lb_map[label]
            else:
                raise NotImplementedError

        im_lb = dict(im=img, lb=label)
        if not self.trans_func is None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        img, label = im_lb['im'], im_lb['lb']
        return img.detach(), label.unsqueeze(0).detach()

    def __len__(self):
        return len(self.img_paths)

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

    ds = CityScapesMini(datapth, annpath, trans_func=trans_func)

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
