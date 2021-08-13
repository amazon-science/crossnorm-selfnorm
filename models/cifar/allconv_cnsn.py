# Code is adapted from https://github.com/google-research/augmix/blob/master/models/cifar/allconv.py
# which is originally licensed under Apache-2.0.

"""AllConv implementation (https://arxiv.org/abs/1412.6806)."""
import math
import torch
import torch.nn as nn
from ..cnsn import CrossNorm, SelfNorm, CNSN
import numpy as np


class GELU(nn.Module):

  def forward(self, x):
    return torch.sigmoid(1.702 * x) * x


def make_layers_custom(cfg, norm_func, pos, beta, crop, cnsn_type):
  """Create a single layer."""
  layers = []
  in_channels = 3
  pos = int(pos)
  print('pos in [conv, norm, relu]: {}'.format(pos))
  assert pos in [1, 2, 3]
  assert cnsn_type in ['sn', 'cn', 'cnsn']

  for v in cfg:
    if v == 'Md':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2), nn.Dropout(p=0.5)]
    elif v == 'A':
      layers += [nn.AvgPool2d(kernel_size=8)]
    elif v == 'NIN':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=1)
      tmp_layers = [conv2d, norm_func(in_channels), GELU()]

      if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
      else:
        crossnorm = None

      if 'sn' in cnsn_type:
        print('using SelfNorm')
        selfnorm = SelfNorm(in_channels)
      else:
        selfnorm = None

      cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

      tmp_layers.insert(pos, cnsn)

      layers += tmp_layers
    elif v == 'nopad':
      conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=0)
      tmp_layers = [conv2d, norm_func(in_channels), GELU()]

      if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
      else:
        crossnorm = None

      if 'sn' in cnsn_type:
        print('using SelfNorm')
        selfnorm = SelfNorm(in_channels)
      else:
        selfnorm = None

      cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

      tmp_layers.insert(pos, cnsn)

      layers += tmp_layers
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      tmp_layers = [conv2d, norm_func(v), GELU()]

      if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
      else:
        crossnorm = None

      if 'sn' in cnsn_type:
        print('using SelfNorm')
        selfnorm = SelfNorm(v)
      else:
        selfnorm = None

      cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

      tmp_layers.insert(pos, cnsn)

      layers += tmp_layers
      in_channels = v

  return nn.Sequential(*layers)


class AllConvNet(nn.Module):
  """AllConvNet main class."""

  def __init__(self, num_classes,
               active_num=None, pos=None, beta=None,
               crop=None, cnsn_type=None):
    super(AllConvNet, self).__init__()

    norm_func = nn.BatchNorm2d

    if beta is not None:
      print('beta: {}'.format(beta))

    if crop is not None:
      print('crop mode: {}'.format(crop))

    self.num_classes = num_classes
    self.width1, w1 = 96, 96
    self.width2, w2 = 192, 192

    self.features = make_layers_custom(
      [w1, w1, w1, 'Md', w2, w2, w2, 'Md', 'nopad', 'NIN', 'NIN', 'A'], norm_func,
       pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type)

    self.classifier = nn.Linear(self.width2, num_classes)

    self.cn_modules = []
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
      elif isinstance(m, CrossNorm):
          self.cn_modules.append(m)

    if 'cn' in cnsn_type:
        self.cn_num = len(self.cn_modules)
        assert self.cn_num > 0
        print('cn_num: {}'.format(self.cn_num))
        self.active_num = active_num
        assert self.active_num > 0
        print('active_num: {}'.format(self.active_num))

  def _enable_cross_norm(self):
      active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
      assert len(set(active_cn_idxs)) == self.active_num
      # print('active_cn_idxs: {}'.format(active_cn_idxs))
      for idx in active_cn_idxs:
          self.cn_modules[idx].active = True

  def forward(self, x, aug=False):
    if aug:
      # print('forward cross norm...')
      # exit()
      self._enable_cross_norm()

    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x
