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

import copy

import torchfcn
import torch.nn as nn
import matplotlib.pyplot as plt
import pdb


def cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss


def compute_pseudo_target(score, prior):
    softmax = nn.Softmax2d()
    probs = softmax(score)

    n, c, h, w = probs.size()
    if prior is not None:
        prior_mat = prior.view(1, c, 1, 1).repeat(n, 1, h, w).float().cuda()
        probs = probs * Variable(prior_mat)
    probs_t = probs.permute(1, 0, 2, 3)
    sum_probs = probs_t.view(c,-1).sum(-1) # for minibatches of size larger than one,
    # we use pixels of all images to compute the class assignment statistics
    norm_factor = torch.sqrt(sum_probs)
    norm_factor = norm_factor.view(1, c, 1, 1).repeat(n, 1, h, w)
    pseudo_gt = probs / norm_factor
    z = pseudo_gt.sum(1).unsqueeze(1).repeat(1, c, 1, 1)
    pseudo_gt = pseudo_gt / z
    return pseudo_gt


class Trainer_siam(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, train_loader_nolbl, val_loader, out, max_iter, prior=None,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model, self.merge = model
        self.optim = optimizer
        self.prior = prior

        self.train_loader_nolbl = train_loader_nolbl
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Europe/London'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):

            if self.cuda:
                data_1, target_1 = data_1.cuda(), target_1.cuda()
            data_1, target_1 = Variable(data_1), Variable(target_1)

            feats_1 = self.model(data_1)

            for batch_idx_2, (data_2, target_2) in enumerate(self.train_loader):

                if self.cuda:
                    data_2, target_2 = data_2.cuda(), target_2.cuda()
                data_2, target_2 = Variable(data_2), Variable(target_2)
                target = target_1.eq(target_2)

                feats_2 = self.model(data_2)
                score = self.merge(feats_1, feats_2)
                target = F.upsample_nearest(target, score.size()[2:])

                loss = cross_entropy2d(score, target, size_average=self.size_average)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss.data[0]) / len(data)

                imgs_1 = data_1.data.cpu()
                imgs_2 = data_2.data.cpu()
                lbls_1 = target_1.data.cpu()
                lbls_2 = target_2.data.cpu()
                lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
                lbl_true = target.data.cpu()

        val_loss /= len(self.val_loader)
        print('\nval_loss = {}\n'.format(val_loss))


    def train_epoch(self):

        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        #########################################
        ### An epoch over the fully-labeled data:
        #########################################
        for batch_idx, (data_1, target_1) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration>0:
                self.validate()

            if self.cuda:
                data_1, target_1 = data_1.cuda(), target_1.cuda()
            data_1, target_1 = Variable(data_1), Variable(target_1)

            feats_1 = self.model(data_1)

            for batch_idx_2, (data_2, target_2) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train innerloop epoch=%d' % self.epoch, ncols=80, leave=False):

                if self.cuda:
                    data_2, target_2 = data_2.cuda(), target_2.cuda()
                data_2, target_2 = Variable(data_2), Variable(target_2)
                target = target_1.eq(target_2).long()
                target[target_1 == -1] = -1
                target[target_2 == -1] = -1

                self.optim.zero_grad()

                feats_2 = self.model(data_2)
                score = self.merge(feats_1, feats_2)

                similarity = F.cosine_similarity(feats_1.view(len(data_1),-1), feats_2.view(len(data_1),-1))
                ### OR:
                #dissimilarity = 1 - score.sum()/(score.size()[2] * score.size()[3] * score.max())

                img1 = data_1.cpu().data.numpy()[0].transpose(1,2,0)
                img2 = data_2.cpu().data.numpy()[0].transpose(1,2,0)
                lbl1 = target_1.cpu().data.numpy()[0]
                lbl2 = target_2.cpu().data.numpy()[0]
                lbleq = target.cpu().data.numpy()[0]

                if (np.squeeze(similarity.cpu().data.numpy()) < 0.5):
                    continue

                pdb.set_trace()

                upscore = F.upsample(score, target.size()[1:], mode='bilinear')

                loss = cross_entropy2d(upscore, target, size_average=self.size_average)
                loss /= len(data_1)

                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')

                if batch_idx_2==(len(self.train_loader)-1):
                    loss.backward()
                else:
                    loss.backward(retain_graph=True)
                self.optim.step()

            if self.iteration >= self.max_iter:
                break


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
