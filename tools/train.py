#!/usr/bin/python
# -*- encoding: utf-8 -*-

import sys
sys.path.insert(0, '.')
import os
import os.path as osp
import random
import logging
import time
import argparse
import itertools
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

from lib.models import model_factory
from configs import cfg_factory
from lib.cityscapes_mini_cv2 import get_data_loader as get_train_dataloader
from lib.carla_sidewalk_cv2 import get_data_loader as get_carla_data_loader

from lib.ohem_ce_loss import OhemCELoss
from lib.lr_scheduler import WarmupPolyLrScheduler
from lib.meters import TimeMeter, AvgMeter
from lib.logger import setup_logger, print_log_msg
from tools.evaluate import eval_model

import wandb

# apex
has_apex = True
try:
    from apex import amp, parallel
except ImportError:
    has_apex = False

weather_options = [
    "ClearNoon",
    "MidRainyNoon",
    # "HardRainSunset"
]

towns = [
    'Town01',
    'Town02',
    # 'Town03',
    'Town04',
    'Town05',
    'Town07',
    # 'Town10HD',
]

img_sizes = [256,512,1024]

sensor_heights = [0.5,1.0,1.5]

patterns = []
for town,img_size,weather,sensor_h in itertools.product(towns,img_sizes,weather_options,sensor_heights):
        pattern = f"{town}_{img_size}_{weather}_{sensor_h}"
        patterns[pattern] = "datasets/carla/sidewalk_"+pattern+"_*"

## fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True
#  torch.backends.cudnn.benchmark = True
#  torch.multiprocessing.set_sharing_strategy('file_system')


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument('--local_rank', dest='local_rank', type=int, default=-1,)
    parse.add_argument('--port', dest='port', type=int, default=44554,)
    parse.add_argument('--model', dest='model', type=str, default='bisenetv2',)
    parse.add_argument('--finetune-from', type=str, default=None,)
    return parse.parse_args()

args = parse_args()
cfg = cfg_factory[args.model]



def set_model():
    net = model_factory[cfg.model_type](8)
    if not args.finetune_from is None:
        net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    if cfg.use_sync_bn: net = set_syncbn(net)
    net.cuda()
    net.train()
    criteria_pre = OhemCELoss(0.7)
    criteria_aux = [OhemCELoss(0.7) for _ in range(cfg.num_aux_heads)]
    return net, criteria_pre, criteria_aux

def set_syncbn(net):
    if has_apex:
        net = parallel.convert_syncbn_model(net)
    else:
        net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    return net


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': cfg.lr_start * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': cfg.lr_start * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=cfg.lr_start,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    return optim


def set_model_dist(net):
    if has_apex:
        net = parallel.DistributedDataParallel(net, delay_allreduce=True)
    else:
        local_rank = dist.get_rank()
        net = nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank, ],
            output_device=local_rank)
    return net


def set_meters():
    time_meter = TimeMeter(cfg.max_iter)
    loss_meter = AvgMeter('loss')
    loss_pre_meter = AvgMeter('loss_prem')
    loss_aux_meters = [AvgMeter('loss_aux{}'.format(i))
            for i in range(cfg.num_aux_heads)]
    return time_meter, loss_meter, loss_pre_meter, loss_aux_meters


def train():
    logger = logging.getLogger()
    is_dist = dist.is_initialized()

    ## dataset
    dl = get_train_dataloader(
        cfg.im_root, cfg.train_im_anns,
        cfg.ims_per_gpu, cfg.scales, cfg.cropsize,
        cfg.max_iter, mode='train', distributed=is_dist)

    valid_dls = dict()
    for set_name, set_pattern in patterns.items():
        carla_dl = get_carla_data_loader(
            set_pattern,
            ims_per_gpu=2,
            scales=None,
            cropsize=None,
            mode='val',
            distributed=is_dist
        )
        valid_dls[set_name] = carla_dl
    cityscapes_valid_dl = get_train_dataloader(cfg.im_root, './datasets/citysc_val.txt', 2, None, None, mode='val', distributed=is_dist)
    valid_dls['cityscapes']=cityscapes_valid_dl
    gta_valid_dl = get_train_dataloader(cfg.im_root, './datasets/gta_val.txt', 2, None, None, mode='val', distributed=is_dist)
    valid_dls['gta']=gta_valid_dl

    ## model
    net, criteria_pre, criteria_aux = set_model()

    if dist.get_rank() == 0:
        exp_name = "cityscapes+gta"
        wandb.init(
            project="bisenet",
            name=exp_name
        )
        wandb.watch(net)

    ## optimizer
    optim = set_optimizer(net)

    ## fp16
    if has_apex:
        opt_level = 'O1' if cfg.use_fp16 else 'O0'
        net, optim = amp.initialize(net, optim, opt_level=opt_level)

    ## ddp training
    net = set_model_dist(net)

    ## meters
    time_meter, loss_meter, loss_pre_meter, loss_aux_meters = set_meters()

    ## lr scheduler
    lr_schdr = WarmupPolyLrScheduler(optim, power=0.9,
        max_iter=cfg.max_iter, warmup_iter=cfg.warmup_iters,
        warmup_ratio=0.1, warmup='exp', last_epoch=-1,)

    ## train loop
    for it, (im, lb) in enumerate(dl):
        net.train()
        im = im.cuda()
        lb = lb.cuda()

        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits, *logits_aux = net(im)
        loss_pre = criteria_pre(logits, lb)
        loss_aux = [crit(lgt, lb) for crit, lgt in zip(criteria_aux, logits_aux)]
        loss = loss_pre + sum(loss_aux)
        if has_apex:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optim.step()
        torch.cuda.synchronize()
        lr_schdr.step()

        time_meter.update()
        loss_meter.update(loss.item())
        loss_pre_meter.update(loss_pre.item())
        _ = [mter.update(lss.item()) for mter, lss in zip(loss_aux_meters, loss_aux)]

        lr = lr_schdr.get_lr()
        lr = sum(lr) / len(lr)
        ## print training log message
        if dist.get_rank() == 0:
            loss_avg = loss_meter.get()[0]
            wandb.log({
                "lr":lr,
                "time":time_meter.get()[0],
                "loss":loss_avg,
                "loss_pre":loss_pre_meter.get()[0],
                **{f"loss_aux_{el.name}":el.get()[0] for el in loss_aux_meters}
            },commit=False)
            if (it + 1) % 100 == 0: print(it,' - ',lr,' - ',loss_avg)
        
            if (it + 1) % 2000 == 0:
                # dump the model and evaluate the result
                save_pth = osp.join(cfg.respth, f"{exp_name}_{it}.pth")
                state = net.module.state_dict()
                torch.save(state, save_pth)
                wandb.save(save_pth)
        if ((it + 1) % 2000 == 0):
            for val_set, val_dl in valid_dls.items():
                logger.info('\nevaluating the model on: '+val_set)
                heads, mious = eval_model(net,val_set,val_dl,it)
                logger.info(tabulate([mious, ], headers=heads, tablefmt='orgtbl'))
            if (dist.get_rank() == 0): wandb.log({k:v for k,v in zip(heads,mious)},commit=False)
        if (dist.get_rank() == 0):
            wandb.log({"t":it},step=it)
    return


def main():
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
        backend='nccl',
        init_method='tcp://127.0.0.1:{}'.format(args.port),
        world_size=torch.cuda.device_count(),
        rank=args.local_rank
    )
    if not osp.exists(cfg.respth): os.makedirs(cfg.respth)
    setup_logger('{}-train'.format(cfg.model_type), cfg.respth)
    train()


if __name__ == "__main__":
    main()
