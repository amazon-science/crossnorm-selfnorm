# Code is adapted from https://github.com/google-research/augmix/blob/master/third_party/WideResNet_pytorch/wideresnet.py
# which is originally licensed under MIT.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..cnsn import CrossNorm, SelfNorm, CNSN


class BasicBlockCustom(nn.Module):
  """Basic ResNet block."""

  def __init__(self, in_planes, out_planes, stride, norm_func,
               pos, beta, crop, cnsn_type, drop_rate=0.0):
    super(BasicBlockCustom, self).__init__()
    # self.bn1 = nn.BatchNorm2d(in_planes)
    self.bn1 = norm_func(in_planes)
    self.relu1 = nn.ReLU(inplace=True)
    self.conv1 = nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)
    # self.bn2 = nn.BatchNorm2d(out_planes)
    self.bn2 = norm_func(out_planes)
    self.relu2 = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(
        out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
    self.drop_rate = drop_rate
    self.is_in_equal_out = (in_planes == out_planes)
    self.conv_shortcut = (not self.is_in_equal_out) and nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=0,
        bias=False) or None

    assert cnsn_type in ['sn', 'cn', 'cnsn']

    if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
    else:
        crossnorm = None

    if 'sn' in cnsn_type:
        print('using SelfNorm')
        if pos == 'pre' and not self.is_in_equal_out:
            selfnorm = SelfNorm(in_planes)
        else:
            selfnorm = SelfNorm(out_planes)
    else:
        selfnorm = None

    self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

    print('pos: {}'.format(pos))
    assert pos in ['residual', 'identity', 'pre', 'post']
    self.pos = pos

  def forward(self, x):

    if not self.is_in_equal_out:
        x = self.relu1(self.bn1(x))

    if self.pos == 'pre':
        out = self.cnsn(x)
    else:
        out = x

    if self.is_in_equal_out:
        out = self.relu1(self.bn1(out))

    out = self.relu2(self.bn2(self.conv1(out)))

    if self.drop_rate > 0:
      out = F.dropout(out, p=self.drop_rate, training=self.training)
    out = self.conv2(out)

    if not self.is_in_equal_out:
      x = self.conv_shortcut(x)

    if self.pos == 'residual':
        out = self.cnsn(out)
    elif self.pos == 'identity':
        x = self.cnsn(x)

    out = torch.add(x, out)

    if self.pos == 'post':
        return self.cnsn(out)
    else:
        return out


class NetworkBlockCustom(nn.Module):
  """Layer container for blocks."""

  def __init__(self,
               nb_layers,
               in_planes,
               out_planes,
               block,
               stride,
               norm_func,
               pos,
               beta,
               crop,
               cnsn_type,
               drop_rate=0.0):
    super(NetworkBlockCustom, self).__init__()
    self.layer = self._make_layer(block, in_planes, out_planes, nb_layers,
                                  stride, norm_func, pos=pos, beta=beta,
                                  crop=crop, cnsn_type=cnsn_type,
                                  drop_rate=drop_rate)

  def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, norm_func,
                  pos, beta, crop, cnsn_type, drop_rate):
    layers = []
    for i in range(nb_layers):
      layers.append(
          block(i == 0 and in_planes or out_planes, out_planes,
                i == 0 and stride or 1, norm_func=norm_func,
                pos=pos, beta=beta, crop=crop,
                cnsn_type=cnsn_type, drop_rate=drop_rate))
    return nn.Sequential(*layers)

  def forward(self, x):
    return self.layer(x)


class WideResNet(nn.Module):
  """WideResNet class."""

  def __init__(self, depth, num_classes, widen_factor=1, drop_rate=0.0,
               active_num=None, pos=None, beta=None, crop=None, cnsn_type=None):
    super(WideResNet, self).__init__()
    norm_func = nn.BatchNorm2d
    n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
    assert (depth - 4) % 6 == 0
    n = (depth - 4) // 6

    # 1st conv before any network block
    self.conv1 = nn.Conv2d(
        3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)

    # 1st block
    self.block1 = NetworkBlockCustom(n, n_channels[0], n_channels[1],
                                     BasicBlockCustom, 1, norm_func,
                                     pos=pos, beta=beta, crop=crop,
                                     cnsn_type=cnsn_type, drop_rate=drop_rate)

    # 2nd block
    self.block2 = NetworkBlockCustom(n, n_channels[1], n_channels[2],
                                     BasicBlockCustom, 2, norm_func,
                                     pos=pos, beta=beta, crop=crop,
                                     cnsn_type=cnsn_type, drop_rate=drop_rate)


    self.block3 = NetworkBlockCustom(n, n_channels[2], n_channels[3],
                                     BasicBlockCustom, 2, norm_func,
                                     pos=pos, beta=beta, crop=crop,
                                     cnsn_type=cnsn_type, drop_rate=drop_rate)


    # global average pooling and classifier
    # self.bn1 = nn.BatchNorm2d(n_channels[3])
    self.bn1 = norm_func(n_channels[3])
    self.relu = nn.ReLU(inplace=True)
    self.fc = nn.Linear(n_channels[3], num_classes)
    self.n_channels = n_channels[3]

    self.cn_modules = []
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear) and m.bias is not None:
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
      for idx in active_cn_idxs:
          self.cn_modules[idx].active = True

  def forward(self, x, aug=False):

      if aug:
          self._enable_cross_norm()

      # stage 0
      out = self.conv1(x)

      # stage 1
      out = self.block1(out)

      # stage 2
      out = self.block2(out)

      # stage 3
      out = self.block3(out)

      out = self.relu(self.bn1(out))
      out = F.avg_pool2d(out, 8)
      out = out.view(out.size(0), -1)
      out = self.fc(out)

      return out
