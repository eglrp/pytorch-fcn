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


class wTrainer(object):
    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out,
                 max_iter,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        #########################################
        # Conv features are not trained for weakly supervised
        for param in model.features.parameters():
            param.requires_grad = False
        #########################################

        self.model = model
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

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data, target, tags) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target, tags = data.cuda(), target.cuda(), tags.cuda()
            data, target, tags = Variable(data, volatile=True), Variable(target), Variable(tags)
            score, score_w = self.model(data)


            if n_class==21:
                upscore_w = F.upsample(score_w, score.size()[-2:], mode='bilinear')
                #score_normmax_factor = torch.max(torch.max(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
                #score_normmin_factor = torch.min(torch.min(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
                #score_norm_w = (upscore_w.data - score_normmin_factor) / score_normmax_factor
                upscore_w_max_val, upscore_w_max_lbl = torch.max(upscore_w, dim=1)
                lbl_pred = upscore_w_max_lbl.cpu().data.numpy()  # For VOC, it includes only FG classes {0,...,19}, so we add 1 to them:
                lbl_pred_bg = lbl_pred + 1
                upscore_w_max_val = upscore_w_max_val.cpu().data.numpy()
                lbl_pred_bg[upscore_w_max_val<.8] = 0
            else:
                #score_normmax_factor = torch.max(torch.max(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
                #score_normmin_factor = torch.min(torch.min(upscore_w.data, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0] + 1e-10
                #score_norm = (upscore_w.data - score_normmin_factor) / score_normmax_factor
                score_max_val, score_max_lbl = torch.max(score, dim=1)
                lbl_pred = score_max_lbl.cpu().data.numpy()
                lbl_pred_bg = lbl_pred
                score_max_val = score_max_val.cpu().data.numpy()
                lbl_pred_bg[score_max_val<.8] = 0

            imgs = data.data.cpu()
            lbl_true = target.data.cpu()

            if self.iteration > 50000:
                pdb.set_trace()

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred_bg):
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

        total_loss = 0

        #########################################
        ### An epoch over the fully-labeled data:
        #########################################
        for batch_idx, (data, target, tags) in tqdm.tqdm(
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
                data, target, tags = data.cuda(), target.cuda(), tags.cuda()
            data, target, tags = Variable(data), Variable(target), Variable(tags)

            ##############################
            ### Cross-entropy loss:
            self.optim.zero_grad()
            score, score_w = self.model(data)

            if n_class==21:
                gap_score = F.avg_pool2d(score_w, kernel_size=(score_w.size()[-2], score_w.size()[-1])).squeeze(-1).squeeze(-1)
            else:
                gap_score = F.avg_pool2d(score, kernel_size=(score.size()[-2], score.size()[-1])).squeeze(-1).squeeze(-1)
            gap_score_sig = nn.Sigmoid()(gap_score)
            loss = F.binary_cross_entropy(gap_score_sig, tags, size_average=self.size_average)
            loss /= len(data)
            total_loss += loss.data
            if np.isnan(float(loss.data[0])):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            if self.iteration >= self.max_iter:
                break

        #pdb.set_trace()
        print("\nTotal training loss in this epoch: {}".format(total_loss))


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break