# Code is adapted from https://github.com/google-research/augmix/blob/master/third_party/ResNeXt_DenseNet/models/densenet.py
# which is originally licensed under MIT.

"""DenseNet implementation (https://arxiv.org/abs/1608.06993)."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..cnsn import CrossNorm, SelfNorm, CNSN


class BottleneckCustom(nn.Module):
  """Bottleneck block for DenseNet."""

  def __init__(self, n_channels, growth_rate, norm_func,
               pos, beta, crop, cnsn_type):
    super(BottleneckCustom, self).__init__()
    inter_channels = 4 * growth_rate
    self.bn1 = norm_func(n_channels)
    self.conv1 = nn.Conv2d(
        n_channels, inter_channels, kernel_size=1, bias=False)
    self.bn2 = norm_func(inter_channels)
    self.conv2 = nn.Conv2d(
        inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    assert cnsn_type in ['sn', 'cn', 'cnsn']

    if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
    else:
        crossnorm = None

    if 'sn' in cnsn_type:
        if pos == 'conv1_pre':
            selfnorm = SelfNorm(n_channels)
        elif pos == 'conv1_post':
            selfnorm = SelfNorm(inter_channels)
        else:
            selfnorm = SelfNorm(growth_rate)
    else:
        selfnorm = None

    self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)
    self.pos = pos
    print('{} in dense module: {}'.format(cnsn_type, pos))
    assert pos in ['conv1_pre', 'conv1_post', 'conv2_post']

  def forward(self, x):

    if self.pos == 'conv1_pre':
      x = self.cnsn(x)

    out = self.conv1(F.relu(self.bn1(x)))

    if self.pos == 'conv1_post':
      out = self.cnsn(out)

    out = self.conv2(F.relu(self.bn2(out)))

    if self.pos == 'conv2_post':
      out = self.cnsn(out)

    out = torch.cat((x, out), 1)
    return out


class SingleLayerCustom(nn.Module):
  """Layer container for blocks."""

  def __init__(self, n_channels, growth_rate, norm_func,
               pos, beta, crop, cnsn_type):
    super(SingleLayerCustom, self).__init__()
    self.bn1 = norm_func(n_channels)
    self.conv1 = nn.Conv2d(
        n_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    assert cnsn_type in ['sn', 'cn', 'cnsn']

    if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
    else:
        crossnorm = None

    if 'sn' in cnsn_type:
        print('using SelfNorm')
        if pos == 'conv1_pre':
            selfnorm = SelfNorm(n_channels)
        else:
            selfnorm = SelfNorm(growth_rate)
    else:
        selfnorm = None

    self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

    self.pos = pos
    print('{} in dense module: {}'.format(cnsn_type, pos))
    assert pos in ['conv1_pre', 'conv1_post']

  def forward(self, x):

    if self.pos == 'conv1_pre':
      x = self.cnsn(x)

    out = self.conv1(F.relu(self.bn1(x)))

    if self.pos == 'conv1_post':
      out = self.cnsn(out)

    out = torch.cat((x, out), 1)
    return out


class Transition(nn.Module):
  """Transition block."""

  def __init__(self, n_channels, n_out_channels, norm_func):
    super(Transition, self).__init__()
    self.bn1 = norm_func(n_channels)
    self.conv1 = nn.Conv2d(
        n_channels, n_out_channels, kernel_size=1, bias=False)

  def forward(self, x):
    out = self.conv1(F.relu(self.bn1(x)))
    out = F.avg_pool2d(out, 2)
    return out


class DenseNet(nn.Module):
  """DenseNet main class."""

  def __init__(self, growth_rate, depth, reduction, n_classes, bottleneck,
               active_num=None, pos=None, beta=None, crop=None, cnsn_type=None):
    super(DenseNet, self).__init__()

    norm_func = nn.BatchNorm2d

    if beta is not None:
      print('beta: {}'.format(beta))

    if crop is not None:
      print('crop mode: {}'.format(crop))

    if bottleneck:
      n_dense_blocks = int((depth - 4) / 6)
    else:
      n_dense_blocks = int((depth - 4) / 3)

    n_channels = 2 * growth_rate
    self.conv1 = nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=False)

    # 1st block
    self.dense1 = self._make_dense_cnsn(n_channels, growth_rate, n_dense_blocks,
                                        bottleneck, norm_func, pos=pos,
                                        beta=beta, crop=crop, cnsn_type=cnsn_type)

    n_channels += n_dense_blocks * growth_rate
    n_out_channels = int(math.floor(n_channels * reduction))
    self.trans1 = Transition(n_channels, n_out_channels, norm_func)

    n_channels = n_out_channels
    # 2nd block
    self.dense2 = self._make_dense_cnsn(n_channels, growth_rate, n_dense_blocks,
                                        bottleneck, norm_func, pos=pos,
                                        beta=beta, crop=crop, cnsn_type=cnsn_type)

    n_channels += n_dense_blocks * growth_rate
    n_out_channels = int(math.floor(n_channels * reduction))
    self.trans2 = Transition(n_channels, n_out_channels, norm_func)

    n_channels = n_out_channels
    # 3rd block
    self.dense3 = self._make_dense_cnsn(n_channels, growth_rate, n_dense_blocks,
                                        bottleneck, norm_func, pos=pos,
                                        beta=beta, crop=crop, cnsn_type=cnsn_type)

    n_channels += n_dense_blocks * growth_rate

    self.bn1 = nn.BatchNorm2d(n_channels)
    self.fc = nn.Linear(n_channels, n_classes)

    self.cn_modules = []
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
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

  def _make_dense_cnsn(self, n_channels, growth_rate, n_dense_blocks, bottleneck, norm_func,
                       pos, beta, crop, cnsn_type):
    layers = []
    for _ in range(int(n_dense_blocks)):
      if bottleneck:
        layers.append(BottleneckCustom(n_channels, growth_rate, norm_func,
                                       pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type))
      else:
        layers.append(SingleLayerCustom(n_channels, growth_rate, norm_func,
                                        pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type))
      n_channels += growth_rate
    return nn.Sequential(*layers)

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

    out = self.conv1(x)
    out = self.trans1(self.dense1(out))
    out = self.trans2(self.dense2(out))
    out = self.dense3(out)
    out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
    out = self.fc(out)
    return out


def densenet(growth_rate=12, depth=40, num_classes=10, config=None):
  model = DenseNet(growth_rate, depth, 1., num_classes, False,
                   active_num=config.active_num, pos=config.pos,
                   beta=config.beta, crop=config.crop, cnsn_type=config.cnsn_type)
  return model
