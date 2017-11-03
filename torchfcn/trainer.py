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
    ### either:
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    ### or:
    #target = target.unsqueeze(1)
    #target_onehot = torch.zeros(target.size()[0],c).scatter_(1, target.cpu().data, torch.ones(target.size()[0],1)).cuda()
    #loss = - torch.sum(log_p * Variable(target_onehot))
    if size_average:
        loss /= mask.data.sum()
    return loss

def soft_cross_entropy2d(input, target, weight=None, size_average=True):
    # input: (n, c, h, w), target: (n, c, h, w)
    n, c, h, w = input.size()
    # log_p: (n, c, h, w)
    log_p = F.log_softmax(input)
    # log_p: (n*h*w, c)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target: (n*h*w,)
    loss = - torch.sum(log_p * target)
    if size_average:
        loss /= (n*h*w)
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
    norm_factor = sum_probs.view(1, c, 1, 1).repeat(n, 1, h, w)
    pseudo_gt = probs / norm_factor
    z = pseudo_gt.sum(1).unsqueeze(1).repeat(1, c, 1, 1)
    pseudo_gt = pseudo_gt / z
    return pseudo_gt


class Trainer(object):
    def __init__(self, cuda, model, optimizer, optimizer_2,
                 train_loader, train_loader_nolbl, val_loader, out,
                 max_iter, mean_embeddings=None, prior=None,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.optim_2 = optimizer_2
        self.mean_embeddings = mean_embeddings
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
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            score = self.model(data)

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
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'optim_2_state_dict': self.optim_2.state_dict(),
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
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration>0:
                #pdb.set_trace()
                self.validate()

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            if not batch_idx%2:
                ##############################
                ### Cross-entropy loss:
                self.optim.zero_grad()
                #self.optim_2.zero_grad()
                score = self.model(data)
                loss = cross_entropy2d(score, target, size_average=self.size_average)
                loss /= len(data)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while training')
                loss.backward()
                self.optim.step()
            else:
                #continue
                ##############################
                ### Clustering loss:
                self.optim.zero_grad()
                #self.optim_2.zero_grad()
                score = self.model(data)
                pseudo_target = compute_pseudo_target(score, self.prior)


                conf, ps = pseudo_target.data.max(1)
                ps[conf<.9] = -1
                if (ps.max()==-1):
                    continue
                target = Variable(ps)


                #loss_2 = soft_cross_entropy2d(score, pseudo_target, size_average=self.size_average)
                loss_2 = cross_entropy2d(score, target, size_average=self.size_average)

                loss_2 /= len(data)
                if np.isnan(float(loss_2.data[0])):
                    raise ValueError('loss is nan while training')
                loss_2.backward()
                #self.optim_2.step()
                self.optim.step()

            if self.iteration >= self.max_iter:
                break


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break