import datetime
import math
import os
import os.path as osp
import shutil

import fcn
import numpy as np
import pytz
import scipy.misc
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import tqdm
import argparse

import copy

import torchfcn
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
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
    # ---- OR ---:
    """
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2012ClassSeg(root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)
    """

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
    SIFTFLOW = 0
    ## If both the above are zero, VOC data is chosen by default.

    if CAMVID:
        train_loader = train_loader_camvid
        val_loader = test_loader_camvid
    elif SIFTFLOW:
        train_loader = train_loader_siftflow
        val_loader = val_loader_siftflow

    model = torchfcn.models.FCN32s(n_class=len(train_loader.dataset.class_names))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])

    #########################################
    #########################################
    #########################################

    if cuda:
        model = model.cuda()

    model.eval()
    pgts = []
    pgts_conf = []

    for batch_idx, (data, target, tags) in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), desc='Get training localizations', ncols=80, leave=False):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        score, score_w = model(data)
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        bad_lbls = np.unique(lbl_pred)
        ok_lbls = np.unique(target.numpy())
        for l in bad_lbls:
            if not l in ok_lbls:
                lbl_pred[lbl_pred==l] = -1
        pgts.append(lbl_pred.astype(np.int8))
        #pgts_conf.append((100*upscore_w_max_val).astype(np.int8))

    pickle.dump(pgts, open('pgts.p', 'w'))

if __name__ == '__main__':
    main()