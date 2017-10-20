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
import pdb
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=150000,
        lr=1.0e-11,
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
        Merge
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


class Merge(nn.Module):
    def __init__(self):
        super(Merge, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(4096, 2, 1),
        )
    def forward(self, x1, x2):
        y = torch.sum(x1 * x2, 1)
        y = torch.unsqueeze(y, 1)
        #y = self.classifier(y)
        #pdb.set_trace()
        return y


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
        batch_size=1, shuffle=True, **kwargs)

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


    # 2. model

    n_class = len(train_loader.dataset.class_names)
    model = torchfcn.models.FCN32s(n_class=n_class)
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)

    #################################
    ### Dropping the score and up_score (and the last Dropout) layers of FCN32
    layers = list(model.children())
    layers = layers[:-4]
    model = nn.Sequential(*layers)
    #################################
    merge = Merge()

    if cuda:
        model = model.cuda()
        merge = merge.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True), 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            {'params': get_parameters(merge, bias=False), 'lr': cfg['lr'] * 10},
            {'params': get_parameters(merge, bias=True), 'lr': cfg['lr'] * 10, 'weight_decay': 0}],
        lr=cfg['lr'],  momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = torchfcn.Trainer_siam(
        cuda=cuda,
        model=(model,merge),
        optimizer=optim,
        train_loader=train_loader_siftflow,
        train_loader_nolbl=train_loader_nolbl,
        val_loader=val_loader_siftflow,
        out=out,
        max_iter=cfg['max_iteration'],
        prior=None,
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
