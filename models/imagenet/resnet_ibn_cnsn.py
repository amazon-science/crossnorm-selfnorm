# Code is adapted from https://github.com/XingangPan/IBN-Net/blob/8efba2b20acf1f891386bfd2f8ffb5d69c491c6a/ibnnet/resnet_ibn.py
# which is originally licensed under MIT.

import math
import warnings

import torch
import torch.nn as nn
import numpy as np
from ..cnsn import CrossNorm, SelfNorm, CNSN

__all__ = ['ResNet_IBN', 'resnet50_ibn_a', 'resnet101_ibn_a', 'resnet152_ibn_a',
           'resnet50_ibn_b', 'resnet101_ibn_b', 'resnet152_ibn_b']


model_urls = {
    'resnet50_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'resnet101_ibn_a': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'resnet50_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_b-9ca61e85.pth',
    'resnet101_ibn_b': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_b-c55f6dba.pth',
}


class IBN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`
    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """
    def __init__(self, planes, ratio=0.5):
        super(IBN, self).__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        # print('excuting ibn with half: {}'.format(self.half))
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out


class BottleneckCustom(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, pos, beta, crop, cnsn_type,
                 ibn=None, stride=1, downsample=None):
        super(BottleneckCustom, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        if ibn == 'a':
            self.bn1 = IBN(planes)
        else:
            self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.IN = nn.InstanceNorm2d(planes * 4, affine=True) if ibn == 'b' else None
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if self.IN is not None and pos == 'post':
            self.cnsn = None
        else:
            assert cnsn_type in ['sn', 'cn', 'cnsn']

            if 'cn' in cnsn_type:
                print('using CrossNorm with crop: {}'.format(crop))
                crossnorm = CrossNorm(crop=crop, beta=beta)
            else:
                crossnorm = None

            if 'sn' in cnsn_type:
                print('using SelfNorm')
                if pos == 'pre':
                    selfnorm = SelfNorm(inplanes)
                else:
                    selfnorm = SelfNorm(planes * self.expansion)
            else:
                selfnorm = None

            self.cnsn = CNSN(crossnorm=crossnorm, selfnorm=selfnorm)

        self.pos = pos
        if pos is not None:
            print('{} in residual module: {}'.format(cnsn_type, pos))
            assert pos in ['residual', 'pre', 'post', 'identity']

    def forward(self, x):
        identity = x

        if self.pos == 'pre':
            x = self.cnsn(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.pos == 'residual':
            out = self.cnsn(out)
        elif self.pos == 'identity':
            identity = self.cnsn(identity)

        out += identity

        if self.IN is not None:
            out = self.IN(out)
        elif self.pos == 'post':
            out = self.cnsn(out)

        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 layers,
                 ibn_cfg=('a', 'a', 'a', None),
                 num_classes=1000, active_num=None, pos=None, beta=None,
                 crop=None, cnsn_type=None):
        self.inplanes = 64
        super(ResNet, self).__init__()
        print('ResNet with ibn, selfnorm and crossnorm...')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if beta is not None:
            print('beta: {}'.format(beta))

        if crop is not None:
            print('crop mode: {}'.format(crop))


        self.layer1 = self._make_layer_custom(BottleneckCustom, 64, layers[0],
                                              pos=pos, beta=beta,
                                              crop=crop, cnsn_type=cnsn_type,
                                              ibn=ibn_cfg[0])

        self.layer2 = self._make_layer_custom(BottleneckCustom, 128, layers[1],
                                              pos=pos, beta=beta,
                                              crop=crop, cnsn_type=cnsn_type,
                                              stride=2, ibn=ibn_cfg[1])

        self.layer3 = self._make_layer_custom(BottleneckCustom, 256, layers[2],
                                              pos=pos, beta=beta,
                                              crop=crop, cnsn_type=cnsn_type,
                                              stride=2, ibn=ibn_cfg[2])

        self.layer4 = self._make_layer_custom(BottleneckCustom, 512, layers[3],
                                              pos=pos, beta=beta,
                                              crop=crop, cnsn_type=cnsn_type,
                                              stride=2, ibn=ibn_cfg[3])

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * BottleneckCustom.expansion, num_classes)

        self.cn_modules = []
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, CrossNorm):
                self.cn_modules.append(m)

        if cnsn_type is not None and 'cn' in cnsn_type:
            self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))

    def _make_layer_custom(self, block, planes, blocks, pos, beta,
                           crop, cnsn_type, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pos=pos, beta=beta,
                            crop=crop, cnsn_type=cnsn_type,
                            ibn=None if ibn == 'b' else ibn,
                            stride=stride, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, pos=pos, beta=beta,
                                crop=crop, cnsn_type=cnsn_type,
                                ibn=None if (ibn == 'b' and i < blocks-1) else ibn))

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

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50_ibn_a(config):
    """Constructs a ResNet-50-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        layers=[3, 4, 6, 3],
        ibn_cfg=('a', 'a', 'a', None),
        active_num=config.active_num,
        pos=config.pos, beta=config.beta,
        crop=config.crop,
        cnsn_type=config.cnsn_type)
    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_a']))
    return model


def resnet101_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_a']))
    return model


def resnet152_ibn_a(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-a model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('a', 'a', 'a', None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-a!")
    return model


def resnet50_ibn_b(config):
    """Constructs a ResNet-50-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(
        layers=[3, 4, 6, 3],
        ibn_cfg=('b', 'b', None, None),
        active_num=config.active_num,
        pos=config.pos, beta=config.beta,
        crop=config.crop, cnsn_type=config.cnsn_type)

    # if pretrained:
    #     model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet50_ibn_b']))
    return model


def resnet101_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-101-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 4, 23, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        model.load_state_dict(torch.hub.load_state_dict_from_url(model_urls['resnet101_ibn_b']))
    return model


def resnet152_ibn_b(pretrained=False, **kwargs):
    """Constructs a ResNet-152-IBN-b model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_IBN(block=Bottleneck_IBN,
                       layers=[3, 8, 36, 3],
                       ibn_cfg=('b', 'b', None, None),
                       **kwargs)
    if pretrained:
        warnings.warn("Pretrained model not available for ResNet-152-IBN-b!")
    return model