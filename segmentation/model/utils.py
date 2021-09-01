import numpy as np
import os
import math
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
from time import strftime
from collections import defaultdict
from six import iteritems
import json
import shutil
# import torch.optim as optim
import torch.nn.functional as F
import functools
import pdb
# from torchvision.utils import make_grid, save_image

def calc_ins_mean_std(x, eps=1e-11):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = x.size()
    # print('x size: {}'.format(x.size()))
    assert (len(size) == 4)
    N, C = size[:2]
    var = x.contiguous().view(N, C, -1).var(dim=2) + eps
    std = var.sqrt().view(N, C, 1, 1)
    mean = x.contiguous().view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return mean, std


def instance_norm_mix(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_ins_mean_std(style_feat)
    content_mean, content_std = calc_ins_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def cn_rand_bbox_v2(size, beta, bbx_thres):
    W = size[2]
    H = size[3]
    while True:
        ratio = np.random.beta(beta, beta)
        cut_rat = np.sqrt(ratio)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        ratio = float(bbx2 - bbx1) * (bby2 - bby1) / (W * H)
        if ratio > bbx_thres:
            break

    return bbx1, bby1, bbx2, bby2


def instance_norm_mix_v2(x1, x2):
    assert x1.size()[:2] == x2.size()[:2]
    x1_mean, x1_std = calc_ins_mean_std(x1)
    x2_mean, x2_std = calc_ins_mean_std(x2)

    x1_norm = (x1 - x1_mean.expand_as(x1)) / x1_std.expand_as(x1)
    x2_norm = (x2 - x2_mean.expand_as(x2)) / x2_std.expand_as(x2)
    x1_x2 = x1_norm * x2_std.expand_as(x1) + x2_mean.expand_as(x1)
    x2_x1 = x2_norm * x1_std.expand_as(x2) + x1_mean.expand_as(x2)

    return x1_x2, x2_x1


# this class aims to combine multi-types cross norm
class CrossNormComb(nn.Module):
    def __init__(self, cn_type=None, beta=None, bbx_thres_1=None,
                 bbx_thres_2=None, lam_1=None, lam_2=None, way=None, crop=None):
        super(CrossNormComb, self).__init__()

        self.active = False
        self.cn_type = cn_type
        
        self.cn_op = functools.partial(cn_op_2ins_space_chan, beta=beta,
                                            crop=crop, bbx_thres=bbx_thres_2,
                                            lam=lam_2, chan=False)
        
    def forward(self, x):
        #if self.training and self.active:
        if self.active and self.training:
            x = self.cn_op(x)
        self.active = False

        return x


def cn_op_2ins_space_chan(x, beta=None, crop='neither', bbx_thres=None, lam=None, chan=False):
    assert crop in ['neither', 'style', 'content', 'both']
    ins_idxs = torch.randperm(x.size()[0]).cuda()
    # print('cn_op_2ins_space_chan with beta: {}, crop: {}， lam: {}, bbx_thres: {}, chan: {}'
    #       .format(beta, crop, lam, bbx_thres, chan))
    # exit()

    if crop in ['style', 'both']:
        bbx3, bby3, bbx4, bby4 = cn_rand_bbox_v2(x.size(), beta=beta, bbx_thres=bbx_thres)
        # print('crop regions for style features')
        x2 = x[ins_idxs, :, bbx3:bbx4, bby3:bby4]
        # print('x2 size: {}'.format(x2.size()))
    else:
        x2 = x[ins_idxs]

    if chan:
        chan_idxs = torch.randperm(x.size()[1]).cuda()
        x2 = x2[:, chan_idxs, :, :]

    if crop in ['content', 'both']:
        x_aug = torch.zeros_like(x)
        # print('crop regions for content features')
        # tmp = torch.arange(x.size()[0]).cuda()
        # x1 = x[:, :, bbx1:bbx2, bby1:bby2]
        # print('x1 size: {}'.format(x1.size()))
        bbx1, bby1, bbx2, bby2 = cn_rand_bbox_v2(x.size(), beta=beta, bbx_thres=bbx_thres)
        x_aug[:, :, bbx1:bbx2, bby1:bby2] = instance_norm_mix(content_feat=x[:, :, bbx1:bbx2, bby1:bby2],
                                                              style_feat=x2)
        mask = torch.ones(x.size()[2:], requires_grad=False).cuda()
        mask[bbx1:bbx2, bby1:bby2] = 0.
        x_aug = x * mask.unsqueeze(0).unsqueeze(1).expand_as(x) + x_aug
    else:
        # x1 = x
        x_aug = instance_norm_mix(content_feat=x, style_feat=x2)
    # x1_aug = instance_norm_mix(content_feat=x1, style_feat=x2)

    if lam is not None:
        # print('using lam', lam)
        x = x * lam + x_aug * (1-lam)
        # print('instance_norm_mix_op interpolate x1 with lam: {}'.format(lam))
    else:
        x = x_aug

    return x



    
class SRMLayer(nn.Module):
    def __init__(self, channel):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        # print('srm forward...')
        b, c, _, _ = x.size()

        # Style pooling
        mean, std = calc_ins_mean_std(x, eps=1e-8)
        mean = mean.squeeze(3)
        std = std.squeeze(3)

        # mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        # std = x.view(b, c, -1).std(-1).unsqueeze(-1)

        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)


class SelfNorm(nn.Module):
    def __init__(self, channel):
        super(SelfNorm, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc_mean = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                                  groups=channel)
        self.cfc_std = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                                 groups=channel)
        self.bn_mean = nn.BatchNorm1d(channel)
        self.bn_std = nn.BatchNorm1d(channel)

    def forward(self, x):
        # print('SelfNorm forward...')
        b, c, _, _ = x.size()

        # Style pooling
        mean, std = calc_ins_mean_std(x, eps=1e-11)

        u = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)  # (b, c, 2)

        # Style integration
        z_mean = self.cfc_mean(u)  # (b, c, 1)
        z_mean = self.bn_mean(z_mean)
        f_mean = torch.sigmoid(z_mean)
        f_mean = f_mean.view(b, c, 1, 1)

        z_std = self.cfc_std(u)  # (b, c, 1)
        z_std = self.bn_std(z_std)
        g_std = torch.sigmoid(z_std)
        g_std = g_std.view(b, c, 1, 1)

        # assert len(x.shape) == len(mean.shape)
        return x * g_std.expand_as(x) + mean.expand_as(x) * (f_mean.expand_as(x)-g_std.expand_as(x))


class SNCN(nn.Module):
    def __init__(self, selfnorm, crossnorm):
        super(SNCN, self).__init__()
        self.selfnorm = selfnorm
        self.crossnorm = crossnorm

    def forward(self, x):
        
        if self.crossnorm and self.crossnorm.active:
            # print('using crossnorm...')
            x = self.crossnorm(x)
            #print('SNCN forward')

        if self.selfnorm:
            # print('using selfnorm...')
            x = self.selfnorm(x)
        return x




class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="max"):
        """  
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode 

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out

    @staticmethod
    def spatial_pyramid_pool(previous_conv, levels, mode):
        """
        Static Spatial Pyramid Pooling method, which divides the input Tensor vertically and horizontally
        (last 2 dimensions) according to each level in the given levels and pools its value according to the given mode.
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width and height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(levels)):
            h_kernel = int(math.ceil(previous_conv_size[0] / levels[i]))
            w_kernel = int(math.ceil(previous_conv_size[1] / levels[i]))
            w_pad1 = int(math.floor((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * levels[i] - previous_conv_size[1]) / 2))
            h_pad1 = int(math.floor((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            h_pad2 = int(math.ceil((h_kernel * levels[i] - previous_conv_size[0]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * levels[i] - previous_conv_size[1]) and \
                   h_pad1 + h_pad2 == (h_kernel * levels[i] - previous_conv_size[0])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2, h_pad1, h_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                spp = x.view(num_sample, -1)
            else:
                spp = torch.cat((spp, x.view(num_sample, -1)), 1)

        return spp


