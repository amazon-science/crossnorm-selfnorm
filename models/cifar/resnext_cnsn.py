# Code is adapted from https://github.com/google-research/augmix/blob/master/third_party/ResNeXt_DenseNet/models/resnext.py
# which is originally licensed under MIT.

"""ResNeXt implementation (https://arxiv.org/abs/1611.05431)."""
import math
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from ..cnsn import CrossNorm, SelfNorm, CNSN


class ResNeXtBottleneckCustom(nn.Module):
  """ResNeXt Bottleneck Block type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)."""
  expansion = 4

  def __init__(self,
               inplanes,
               planes,
               cardinality,
               base_width,
               norm_func,
               pos, beta, crop, cnsn_type,
               stride=1,
               downsample=None):
    super(ResNeXtBottleneckCustom, self).__init__()

    dim = int(math.floor(planes * (base_width / 64.0)))

    self.conv_reduce = nn.Conv2d(
        inplanes,
        dim * cardinality,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False)
    self.bn_reduce = norm_func(dim * cardinality)

    self.conv_conv = nn.Conv2d(
        dim * cardinality,
        dim * cardinality,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=cardinality,
        bias=False)
    self.bn = norm_func(dim * cardinality)

    self.conv_expand = nn.Conv2d(
        dim * cardinality,
        planes * 4,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=False)
    self.bn_expand = norm_func(planes * 4)

    self.downsample = downsample

    assert cnsn_type in ['sn', 'cn', 'cnsn']

    if 'cn' in cnsn_type:
        print('using CrossNorm with crop: {}'.format(crop))
        crossnorm = CrossNorm(crop=crop, beta=beta)
    else:
        crossnorm = None

    if 'sn' in cnsn_type:
        print('using SelfNorm')
        if pos in ['pre', 'identity']:
            selfnorm = SelfNorm(inplanes)
        else:
            selfnorm = SelfNorm(planes * 4)
    else:
        selfnorm = None

    self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

    self.pos = pos
    # if pos is not None:
    print('{} in residual module: {}'.format(cnsn_type, pos))
    assert pos in ['residual', 'identity', 'pre', 'post']

  def forward(self, x):
    residual = x

    if self.pos == 'pre':
        x = self.cnsn(x)

    bottleneck = self.conv_reduce(x)
    bottleneck = F.relu(self.bn_reduce(bottleneck), inplace=True)

    bottleneck = self.conv_conv(bottleneck)
    bottleneck = F.relu(self.bn(bottleneck), inplace=True)

    bottleneck = self.conv_expand(bottleneck)
    bottleneck = self.bn_expand(bottleneck)

    if self.pos == 'residual':
        bottleneck = self.cnsn(bottleneck)

    if self.pos == 'identity':
        residual = self.cnsn(residual)

    if self.downsample is not None:
      residual = self.downsample(x)

    x = F.relu(residual + bottleneck, inplace=True)

    if self.pos == 'post':
        x = self.cnsn(x)

    return x


class CifarResNeXt(nn.Module):
  """ResNext optimized for the Cifar dataset, as specified in https://arxiv.org/pdf/1611.05431.pdf."""

  def __init__(self, depth, cardinality, base_width, num_classes,
               active_num=None, pos=None, beta=None,
               crop=None, cnsn_type=None):
    super(CifarResNeXt, self).__init__()

    norm_func = nn.BatchNorm2d

    if beta is not None:
        print('beta: {}'.format(beta))

    if crop is not None:
        print('crop mode: {}'.format(crop))

    # Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
    assert (depth - 2) % 9 == 0, 'depth should be one of 29, 38, 47, 56, 101'
    layer_blocks = (depth - 2) // 9

    self.cardinality = cardinality
    self.base_width = base_width
    self.num_classes = num_classes

    self.conv_1_3x3 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    self.bn_1 = norm_func(64)

    self.inplanes = 64

    # 1st block
    self.stage_1 = self._make_layer_cnsn(ResNeXtBottleneckCustom, 64, layer_blocks, norm_func,
                                         pos=pos, beta=beta, crop=crop,
                                         cnsn_type=cnsn_type, stride=1)

    # 2nd block
    self.stage_2 = self._make_layer_cnsn(ResNeXtBottleneckCustom, 128, layer_blocks, norm_func,
                                         pos=pos, beta=beta, crop=crop,
                                         cnsn_type=cnsn_type, stride=2)

    # 3rd block
    self.stage_3 = self._make_layer_cnsn(ResNeXtBottleneckCustom, 256, layer_blocks, norm_func,
                                         pos=pos, beta=beta, crop=crop,
                                         cnsn_type=cnsn_type, stride=2)


    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(256 * ResNeXtBottleneckCustom.expansion, num_classes)

    self.cn_modules = []
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
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

  def _make_layer_cnsn(self, block, planes, blocks, norm_func, pos,
                       beta, crop, cnsn_type, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
          nn.Conv2d(
              self.inplanes,
              planes * block.expansion,
              kernel_size=1,
              stride=stride,
              bias=False),
          norm_func(planes * block.expansion),
      )

    layers = []
    layers.append(
        block(self.inplanes, planes, self.cardinality, self.base_width, norm_func,
              pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type,
              stride=stride, downsample=downsample))
    self.inplanes = planes * block.expansion
    for _ in range(1, blocks):
      layers.append(
          block(self.inplanes, planes, self.cardinality, self.base_width, norm_func,
                pos=pos, beta=beta, crop=crop, cnsn_type=cnsn_type))

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

    x = self.conv_1_3x3(x)
    x = F.relu(self.bn_1(x), inplace=True)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)


def resnext29(num_classes=10, cardinality=4, base_width=32, config=None):
  model = CifarResNeXt(29, cardinality, base_width, num_classes,
                       active_num=config.active_num, pos=config.pos,
                       beta=config.beta, crop=config.crop, cnsn_type=config.cnsn_type)
  return model
