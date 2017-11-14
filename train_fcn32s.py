#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import numpy as np

import pytz
import torch
import yaml

import torchfcn
from get_mean_embeddings import *
import pickle

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=150000,
        lr=1.0e-6,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=2488,
    )
}


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone('Europe/London'))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    log_dir = osp.join(here, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))


here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir('fcn32s', args.config, cfg)
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset

    root = osp.expanduser('~/bin/')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2012ClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    train_loader_nolbl = torch.utils.data.DataLoader(
        torchfcn.datasets.SBDClassSeg(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)

    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2011ClassSeg(root, split='seg11valid', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    #### SIFT-FLOW
    train_loader_siftflow = torch.utils.data.DataLoader(
        torchfcn.datasets.SiftFlowData(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)

    val_loader_siftflow = torch.utils.data.DataLoader(
        torchfcn.datasets.SiftFlowData(root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    #### CAMVID
    train_loader_camvid = torch.utils.data.DataLoader(
        torchfcn.datasets.CamVid(root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)

    val_loader_camvid = torch.utils.data.DataLoader(
        torchfcn.datasets.CamVid(root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    test_loader_camvid = torch.utils.data.DataLoader(
        torchfcn.datasets.CamVid(root, split='test', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    #########################################
    #########################################
    #########################################
    # 2. model and dataset

    CAMVID = 0
    SIFTFLOW = 1
    ## If both the above are zero, VOC data is chosen by default.

    if CAMVID:
        train_loader = train_loader_camvid
        val_loader = test_loader_camvid
    elif SIFTFLOW:
        train_loader = train_loader_siftflow
        val_loader = val_loader_siftflow

    model = torchfcn.models.FCN32s(n_class=len(train_loader.dataset.class_names))
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        #vgg16 = torchfcn.models.VGG16(pretrained=True)
        #model.copy_params_from_vgg16(vgg16)

        ##npdicts = caffe_to_numpy('./models/deploy_vgg16_imagenet1000.prototxt', './models/vgg16_imagenet1000.caffemodel')
        npdicts = pickle.load(open('saved.p', 'r')) # saved.p is the pickled layer weights of the official pre-trained vgg16 that I extracted using caffe_to_numpy.py
        model.copy_params_from_numpydict(npdicts)

    #########################################
    #########################################
    #########################################

    if cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    #########################################
    ### For weak supervision, use torchfcn.wTrainer
    ### For full supervision, use torchfcn.Trainer
    ### Accordingly, the learning rate should be adjusted (e.g. 1e-5 vs 1e-10)
    trainer = torchfcn.wTrainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
