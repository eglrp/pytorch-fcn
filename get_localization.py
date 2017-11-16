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
    model_att = torchfcn.models.Attention(n_class=len(train_loader.dataset.class_names))
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['model_state_dict'])
    model_att.load_state_dict(checkpoint['model_att_state_dict'])

    #########################################
    #########################################
    #########################################

    if cuda:
        model = model.cuda()
        model_att = model_att.cuda()

    model.eval()
    model_att.eval()

    n_class = len(train_loader.dataset.class_names)

    label_trues, label_preds = [], []

    pgts = []
    pgts_conf = []

    for batch_idx, (data, target, tags) in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader), desc='Get training localizations', ncols=80, leave=False):

        if cuda:
            data, target, tags = data.cuda(), target.cuda(), tags.cuda()
        data, target, tags = Variable(data, volatile=True), Variable(target), Variable(tags)
        score, score_w = model(data)
        score_w = model_att(score_w)

        if n_class==21:
            upscore_w = F.upsample(score_w, score.size()[-2:], mode='bilinear')
            score_normmin_factor = torch.min(torch.min(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
            upscore_w.data = upscore_w.data - score_normmin_factor
            score_normmax_factor = torch.max(torch.max(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
            upscore_w.data = upscore_w.data / score_normmax_factor

            nontags = (1 - tags).data.nonzero()[:, 1]
            upscore_w[:, nontags, :, :] = -100

            upscore_w_max_val, upscore_w_max_lbl = torch.max(upscore_w, dim=1)
            lbl_pred = upscore_w_max_lbl.cpu().data.numpy()  # For VOC, it includes only FG classes {0,...,19}, so we add 1 to them:
            lbl_pred_bg = lbl_pred + 1
            upscore_w_max_val = upscore_w_max_val.data.cpu().numpy()
            upscore_w_max_val = upscore_w_max_val / upscore_w_max_val.max()
            #pdb.set_trace()
            #lbl_pred_bg[upscore_w_max_val<.9] = -1
            lbl_pred_bg[upscore_w_max_val<.7] = 0
        else:
            #score_normmax_factor = torch.max(torch.max(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
            #score_normmin_factor = torch.min(torch.min(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
            #score_norm = (upscore_w.data - score_normmin_factor) / score_normmax_factor
            upscore_w_max_val, score_max_lbl = torch.max(score, dim=1)
            upscore_w_max_val = upscore_w_max_val / upscore_w_max_val.max()
            lbl_pred = score_max_lbl.cpu().data.numpy()
            lbl_pred_bg = lbl_pred

        pgts.append(lbl_pred_bg.astype(np.int8))
        #pgts_conf.append((100*upscore_w_max_val).astype(np.int8))

    pickle.dump(pgts, open('pgts.p', 'w'))

if __name__ == '__main__':
    main()