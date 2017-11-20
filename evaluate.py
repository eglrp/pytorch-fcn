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

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian


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
    val_loader_siftflow = torch.utils.data.DataLoader(
        torchfcn.datasets.SiftFlowData(root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)

    #### CAMVID
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
        val_loader = test_loader_camvid
    elif SIFTFLOW:
        val_loader = val_loader_siftflow

    model = torchfcn.models.FCN32s(n_class=len(val_loader.dataset.class_names))
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
    n_class = len(val_loader.dataset.class_names)
    label_trues, label_preds = [], []
    visualizations = []

    for batch_idx, (data, target, tags) in tqdm.tqdm(
            enumerate(val_loader), total=len(val_loader), desc='Get training localizations', ncols=80, leave=False):
        if cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        score, score_w = model(data)
        ######################
        ### Applying DenseCRF:

        score_forcrf = score[0].cpu().data.numpy()
        #score_forcrf = np.exp(score_forcrf-score_forcrf.max())
        score_forcrf = score_forcrf / score_forcrf.max(axis=0)
        d = dcrf.DenseCRF2D(data.size()[3], data.size()[2], n_class)
        unary = unary_from_softmax(score_forcrf)

        d.setUnaryEnergy(unary)
        img = np.ascontiguousarray(data.cpu().data.numpy()[0].transpose(1,2,0), dtype=np.uint8)
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(3)
        MAP = np.argmax(Q, axis=0)
        lcrf = MAP.reshape((img.shape[0], img.shape[1]))
        lbl_pred = np.expand_dims(lcrf, axis=0)

        """
        bad_lbls = np.arange(n_class)
        ok_lbls = np.unique(target.numpy())
        for l in bad_lbls:
            if not l in ok_lbls:
                score[0, l, :] = 1e-10
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        """

        ####################
        ### Evaluating the localization maps:
        imgs = data[0].data.cpu()
        for lt, lp in zip(target, lbl_pred):
            imgs, lt = val_loader.dataset.untransform(imgs, lt)
            label_trues.append(lt)
            label_preds.append(lp)
            if len(visualizations) < 9:
                viz = fcn.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                visualizations.append(viz)

        pgts.append(lbl_pred.astype(np.int8))
        #pgts_conf.append((100*upscore_w_max_val).astype(np.int8))

    metrics = torchfcn.utils.label_accuracy_score(label_trues, label_preds, n_class)
    print(metrics)
    out = 'visualization_viz_evaluation'
    if not osp.exists(out):
        os.makedirs(out)
    out_file = osp.join(out, 'sample_results.jpg')
    scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

    #pickle.dump(pgts, open('pgts.p', 'w'))

if __name__ == '__main__':
    main()