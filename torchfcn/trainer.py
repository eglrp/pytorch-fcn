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
import pickle


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
    ### either:
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    ### or:
    #target = target.unsqueeze(1)
    #target_onehot = torch.zeros(target.size()[0],c).scatter_(1, target.cpu().data, torch.ones(target.size()[0],1)).cuda()
    #loss = - torch.sum(log_p * Variable(target_onehot))
    if size_average:
        loss /= mask.data.sum()
    return loss


class Trainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out,
                 max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model, _ = model
        self.optim = optimizer

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

        #####################################
        ### Point-wise supervision
        self.NUM_GT_POINTS = 10
        self.PICKED_POINTS_FILE = 'picked_points_{}.p'.format(self.NUM_GT_POINTS)
        self.total_picked_points = []
        if os.path.exists(self.PICKED_POINTS_FILE):
            self.total_picked_points = pickle.load(open(self.PICKED_POINTS_FILE, 'r'))
        #####################################

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target, _) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score, _ = self.model(data)

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while validating')
            val_loss += float(loss.data[0]) / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = fcn.utils.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        metrics = torchfcn.utils.label_accuracy_score(
            label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        scipy.misc.imsave(out_file, fcn.utils.get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = \
                datetime.datetime.now(pytz.timezone('Europe/London')) - \
                self.timestamp_start
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        print('\n iteration {}: mean_iu={}\n'.format(self.iteration, mean_iu))
        print('IOUs: {}\n'.format(metrics[-1]))
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        #########################################
        ### An epoch over the fully-labeled data:
        #########################################
        for batch_idx, (data, target, _) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration>0:
                #pdb.set_trace()
                self.validate()
                self.model.train()

            if target.numpy().max()==-1:
                continue

            #############################################
            ### Point-wise Supervision:
            if 1:
                labels = target.numpy()
                lbl_list = np.unique(labels[labels>=0])

                if not os.path.exists(self.PICKED_POINTS_FILE):
                    labels_points = -np.ones_like(labels)
                    picked_points = []
                    for i,l in enumerate(lbl_list):
                        total_points = np.where(labels==l)
                        indices = np.random.randint(total_points[0].shape[0], size=self.NUM_GT_POINTS)
                        points = zip(total_points[1][indices], total_points[2][indices])
                        picked_points.append(points)
                        for n in range(self.NUM_GT_POINTS):
                            labels_points[0, points[n][0], points[n][1]] = l
                    self.total_picked_points.append(picked_points)
                else:
                    picked_points = self.total_picked_points[batch_idx]
                    labels_points = -np.ones_like(labels)
                    for i,l in enumerate(lbl_list):
                        points = picked_points[i]
                        for n in range(self.NUM_GT_POINTS):
                            labels_points[0, points[n][0], points[n][1]] = l
                target = torch.from_numpy(labels_points.astype(np.int64))
            #############################################

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            ##############################
            ### Cross-entropy loss:
            self.optim.zero_grad()
            score, _ = self.model(data)
            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss /= len(data)
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            if self.iteration >= self.max_iter:
                break

        if not os.path.exists(self.PICKED_POINTS_FILE):
            pickle.dump(self.total_picked_points, open(self.PICKED_POINTS_FILE, 'w'))


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break