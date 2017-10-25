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

def get_mean_embedding(cuda, model, train_loader):
    model.train()

    n_class = len(train_loader.dataset.class_names)

    out_channels = list(model.children())[-1].out_channels  #The dimensionality of the output layer.
    # Here I simply assumed the output dimension to be the same as the number of classes like the baseline FCN32s
    sum_embeddings = torch.zeros((n_class, out_channels))
    class_frequency = torch.zeros((n_class,))

    #########################################
    ### An epoch over the fully-labeled data:
    #########################################
    for batch_idx, (data, target) in tqdm.tqdm(
            enumerate(train_loader), total=len(train_loader),
            desc='Train cluster mean embeddings: ', ncols=80, leave=False):

        target_np = target.numpy()[0]
        data_np = data.numpy()[0].transpose(1,2,0)
        mask = target_np >= 0
        class_frequency_current = np.bincount(target_np[mask], minlength=n_class)

        if cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        score = model(data)

        score_resh = score.permute(1, 0, 2, 3)
        score_resh = score_resh.view(n_class, -1)

        for c in range(n_class):
            if not class_frequency_current[c]:
                continue
            score_resh_masked = score_resh[target.view(1, -1).repeat(n_class, 1) == c].view(n_class, -1)
            sum_embeddings[:,c] = sum_embeddings[:,c] + torch.sum(score_resh_masked, 1).cpu().data

        class_frequency = class_frequency + torch.from_numpy(class_frequency_current).float()

    class_frequency = torch.unsqueeze(class_frequency, 1).repeat(1, out_channels)
    mean_embeddings = sum_embeddings / (class_frequency.transpose(1, 0))

    return mean_embeddings

"""

score_rep = score.view(n_class, 1, -1).repeat(1, self.mean_embeddings.size()[-1], 1)
embed_rep = self.mean_embeddings.view(self.mean_embeddings.size()[0], self.mean_embeddings.size()[1], 1).repeat(1, 1, score_rep.size()[-1])
dist = torch.norm(score_rep-Variable(embed_rep.cuda()), p=2, dim=0)
values, indices = dist.min(dim=0)
lbl_pred = indices.view(target.size()[1],target.size()[2]).cpu().data.numpy()
lbl_pred = np.expand_dims(lbl_pred, 0)

"""