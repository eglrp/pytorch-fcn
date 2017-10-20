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
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models as torchmodels
from tensorflow.contrib.keras.python.keras.utils import to_categorical


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

from scipy.spatial.distance import correlation

class Trainer_siam(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, train_loader_nolbl, val_loader, out, max_iter, prior=None,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model, self.merge = model

        self.model_fixed = torchmodels.resnet101(pretrained=True)
        layers = list(self.model_fixed.children())
        layers = layers[:-1]
        self.model_fixed = nn.Sequential(*layers).cuda()


        self.model.load_state_dict(torch.load('model_siftflow_dict_epoch_1.pth'))
        self.merge.load_state_dict(torch.load('merge_siftflow_dict_epoch_1.pth'))


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

        if osp.exists(self.train_loader.dataset.__class__.__name__ + '_similarity_features.npy'):
            self.features = np.load(self.train_loader.dataset.__class__.__name__ + '_similarity_features.npy').item()
        else:
            self.features = {}
            for batch_idx, (data_1, _, name) in tqdm.tqdm(
                    enumerate(self.train_loader), total=len(self.train_loader),
                    desc='Global features for similarity:', ncols=80, leave=False):
                if self.cuda:
                    data_1 = data_1.cuda()
                data_1 = Variable(data_1)
                self.features[name[0]] = self.model_fixed(data_1).cpu().data.numpy().squeeze()

            for batch_idx, (data_1, _, name) in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Global features for similarity"', ncols=80, leave=False):
                if self.cuda:
                    data_1 = data_1.cuda()
                data_1 = Variable(data_1)
                self.features[name[0]] = self.model_fixed(data_1).cpu().data.numpy().squeeze()
            np.save(self.train_loader.dataset.__class__.__name__ + '_similarity_features.npy', self.features)

    def validate(self):
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (data_1, target_1, name_1) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):

            if self.cuda:
                data_1, target_1 = data_1.cuda(), target_1.cuda()
            data_1, target_1 = Variable(data_1), Variable(target_1)

            img1 = data_1.cpu().data.numpy()[0].transpose(1, 2, 0)
            lbl1 = target_1.cpu().data.numpy()[0]

            feats_1_sim = self.features[name_1[0]]

            feats_1 = self.model(data_1)

            n, h, w = target_1.size()
            pseudo_label = np.zeros((h, w, n_class))

            for batch_idx_2, (data_2, target_2, name_2) in enumerate(self.train_loader):

                feats_2_sim = self.features[name_2[0]]

                dissimilarity = correlation(feats_1_sim, feats_2_sim)

                if (np.squeeze(dissimilarity) > 0.32):
                    continue

                if self.cuda:
                    data_2, target_2 = data_2.cuda(), target_2.cuda()
                data_2, target_2 = Variable(data_2), Variable(target_2)
                target = target_1.eq(target_2).long()

                feats_2 = self.model(data_2)
                score = self.merge(feats_1, feats_2)

                img2 = data_2.cpu().data.numpy()[0].transpose(1,2,0)
                lbl2 = target_2.cpu().data.numpy()[0]
                lbleq = target.cpu().data.numpy()[0]


                upscore = F.upsample(score, target.size()[1:], mode='bilinear')


                lbl_pred = upscore.max(1)[1][0].cpu().data.numpy()
                lbl_pred[lbl2 == -1] = 0
                lbl_pred_rep = np.expand_dims(lbl_pred, -1)
                lbl_pred_rep = lbl_pred_rep.repeat(n_class, -1)

                lbl2oh = to_categorical(lbl2, n_class)
                lbl2oh_resh = lbl2oh.reshape((h, w, n_class))
                pseudo_label += lbl2oh_resh * lbl_pred_rep

                loss = cross_entropy2d(upscore, target, size_average=self.size_average)
                if np.isnan(float(loss.data[0])):
                    raise ValueError('loss is nan while validating')
                val_loss += float(loss.data[0]) / len(data_1)


            pseudo_label_max = np.max(pseudo_label, axis=-1)
            pseudo_label = np.argmax(pseudo_label, axis=-1)
            pseudo_label[pseudo_label_max==0] = -1

            plt.subplot(121)
            plt.imshow(lbl1)
            plt.title('Ground Truth')
            plt.subplot(122)
            plt.imshow(pseudo_label)
            plt.title('Label Transfer Result')
            plt.show()

            pdb.set_trace()

        val_loss /= len(self.val_loader)
        pdb.set_trace()
        print('\nval_loss = {}\n'.format(val_loss))


    def train_epoch(self):

        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        #########################################
        ### An epoch over the fully-labeled data:
        #########################################
        for batch_idx, (data_1, target_1, name_1) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.train_loader)
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0 and self.iteration>0:
                self.validate()

            if self.cuda:
                data_1, target_1 = data_1.cuda(), target_1.cuda()
            data_1, target_1 = Variable(data_1), Variable(target_1)

            feats_1_sim = self.features[name_1[0]]

            feats_1 = self.model(data_1)

            for batch_idx_2, (data_2, target_2, name_2) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train innerloop epoch=%d' % self.epoch, ncols=80, leave=False):

                feats_2_sim = self.features[name_2[0]]

                dissimilarity = correlation(feats_1_sim, feats_2_sim)

                if (np.squeeze(dissimilarity) > 0.315):
                    continue

                if self.cuda:
                    data_2, target_2 = data_2.cuda(), target_2.cuda()
                data_2, target_2 = Variable(data_2), Variable(target_2)
                target = target_1.eq(target_2).long()
                target[target_1 == -1] = -1
                target[target_2 == -1] = -1

                if (target.max().cpu().data.numpy()[0]==-1):
                    continue

                self.optim.zero_grad()

                feats_2 = self.model(data_2)
                score = self.merge(feats_1, feats_2)

                """
                img1 = data_1.cpu().data.numpy()[0].transpose(1,2,0)
                img2 = data_2.cpu().data.numpy()[0].transpose(1,2,0)
                lbl1 = target_1.cpu().data.numpy()[0]
                lbl2 = target_2.cpu().data.numpy()[0]
                lbleq = target.cpu().data.numpy()[0]
                """

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
