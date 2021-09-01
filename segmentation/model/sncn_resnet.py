import torch
import torch.nn as nn
import numpy as np
import pdb
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from .utils import CrossNormComb, SelfNorm, SNCN, SRMLayer

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    

class BasicBlockCustom(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, pos, cn_type, beta, bbx_thres_1, bbx_thres_2, lam_1, lam_2, way, crop, sncn_type, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlockCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        assert sncn_type in ['sn', 'cn', 'sncn', 'srm', 'srmcn']

        if 'cn' in sncn_type:
            print('cn_type: {}'.format(cn_type))
            crossnorm = CrossNormComb(cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1,
                                      bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2,
                                      way=way, crop=crop)
        else:
            crossnorm = None

        if 'sn' in sncn_type:
            print('using SelfNorm module')
            if pos == 'pre' and not self.is_in_equal_out:
                selfnorm = SelfNorm(in_planes)
            else:
                selfnorm = SelfNorm(out_planes)
        else:
            selfnorm = None

        # if using selfnorma and crossnorm at the same time, then use them at the same location
        # separating their positions is not supported currently.
        self.sncn = SNCN(selfnorm=selfnorm, crossnorm=crossnorm)
        
        self.pos = pos
        if pos is not None:
            print('{} in residual module: {}'.format(sncn_type, pos))
            assert pos in ['residual', 'identity', 'pre', 'post']

    
    def forward(self, x):
        identity = x

        if self.pos == 'pre':
            out = self.sncn(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.pos == 'residual':
            out = self.sncn(out)
        elif self.pos == 'identity':
            identity = self.sncn(identity)


        out += identity
        out = self.relu(out)

        if self.pos == 'post':
            return self.sncn(out)
        else:
            return out

    




class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

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

        out += identity
        out = self.relu(out)

        return out

class BottleneckCustom(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, pos, cn_pos, cn_type, beta, bbx_thres_1, bbx_thres_2, lam_1, lam_2, way, crop, sncn_type, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, custom=False):
        super(BottleneckCustom, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.custom = custom
        
        if self.custom:
            assert sncn_type in ['sn', 'cn', 'sncn', 'srm', 'srmcn']

            if 'cn' in sncn_type and cn_pos is None:
                print('cn_type: {}'.format(cn_type))
                crossnorm = CrossNormComb(cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1,
                                          bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2,
                                          way=way, crop=crop)
            else:
                crossnorm = None

            if 'sn' in sncn_type:
                print('using SelfNorm module')
                if pos == 'pre' and self.downsample is None:
                    selfnorm = SelfNorm(in_planes)
                else:
                    selfnorm = SelfNorm(planes * self.expansion)
            elif 'srm' in sncn_type:
                print('using SRMLayer module')
                if pos == 'pre' and not self.is_in_equal_out:
                    selfnorm = SRMLayer(in_planes)
                else:
                    selfnorm = SRMLayer(planes * self.expansion)
            else:
                selfnorm = None

            # if using selfnorm and crossnorm at the same time, then use them at the same location
            # separating their positions is not supported currently.
            self.sncn = SNCN(selfnorm=selfnorm, crossnorm=crossnorm)
            if 'cn' in sncn_type and cn_pos is not None:
                self.real_cn = CrossNormComb(cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1,
                                          bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2,
                                          way=way, crop=crop)
            self.cn_pos = cn_pos


            self.pos = pos
            if pos is not None:
                print('{} in residual module: {}'.format(sncn_type, pos))
                assert pos in ['residual', 'identity', 'pre', 'post']


    def forward(self, x):
        identity = x
        
        out = x
        if self.custom:
            if self.pos == 'pre':
                out = self.sncn(out)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        if self.custom:
            if self.pos == 'residual':
                out = self.sncn(out)
            elif self.pos == 'identity':
                identity = self.sncn(out)
                


        out += identity
        out = self.relu(out)
        
        if self.custom:
            if self.pos == 'post':
                out = self.sncn(out)
            if self.cn_pos == 'post':
                out = self.real_cn(out)
        return out



class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, block_idxs=None, active_num=None, pos=None, cn_type=None, beta=None, bbx_thres_1=None, bbx_thres_2=None, lam_1=None, lam_2=None, way=None, crop=None, affine=None, sncn_type=None, cn_pos=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        self.sncn_type = sncn_type
        if block_idxs:
            block_idxs = block_idxs.split('_')
            block_idxs = list(map(int, block_idxs))
        print('block_idxs: {}'.format(block_idxs))
        print('cn_type: {}'.format(cn_type))
        if affine:
            print('affine: {}'.format(affine))
        if beta is not None:
            print('beta: {}'.format(beta))
        if bbx_thres_1 is not None:
            assert 0 < bbx_thres_1 < 1
            print('bbx_thres_1: {}'.format(bbx_thres_1))
        if bbx_thres_2 is not None:
            assert 0 < bbx_thres_2 < 1
            print('bbx_thres_2: {}'.format(bbx_thres_2))

        if lam_1 is not None:
            print('lam_1: {}'.format(lam_1))
        if lam_2 is not None:
            print('lam_2: {}'.format(lam_2))

        if way is not None:
            assert way in [1, 2]
            print('way: {}'.format(way))
        if crop is not None:
            print('crop in 2 instance mode: {}'.format(crop))


        self.block_idxs = block_idxs
        if block_idxs and 0 in block_idxs:
            self.img_cn = CrossNormComb(cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1,
                                          bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2,
                                          way=way, crop=crop)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block_idxs and 1 in block_idxs:
            self.layer1 = self._make_layer(block, 64, layers[0], pos=pos, cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, cn_pos=cn_pos, custom=True)
        else:
            self.layer1 = self._make_layer(block, 64, layers[0], custom=False)

        if block_idxs and 2 in block_idxs:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], pos=pos, cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, cn_pos=cn_pos, custom=True)
        else:
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], custom=False)
        
        if block_idxs and 3 in block_idxs:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], pos=pos, cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, cn_pos=cn_pos, custom=True)    
        else:
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], custom=False)

        if block_idxs and 4 in block_idxs:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], pos=pos, cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, cn_pos=cn_pos, custom=True)
        else:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], custom=False)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        

        self.cn_modules = []

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()
            elif isinstance(m, CrossNormComb):
                self.cn_modules.append(m)

        if 'cn' in sncn_type or cn_pos is not None:
            self.cn_num = len(self.cn_modules)
            assert self.cn_num > 0
            print('cn_num: {}'.format(self.cn_num))
            self.active_num = active_num
            assert self.active_num > 0
            print('active_num: {}'.format(self.active_num))


        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, pos=None, cn_type=None, beta=None, bbx_thres_1=None, bbx_thres_2=None, lam_1=None, lam_2=None, way=None, crop=None, sncn_type=None, cn_pos=None, stride=1, dilate=False, custom=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pos=pos, cn_type=cn_type, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, cn_pos=cn_pos, stride=stride, downsample=downsample, 
            groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, custom=custom))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, pos=pos, cn_type=cn_type, cn_pos=cn_pos, beta=beta, bbx_thres_1=bbx_thres_1, bbx_thres_2=bbx_thres_2, lam_1=lam_1, lam_2=lam_2, way=way, crop=crop, sncn_type=sncn_type, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, custom=custom))

        return nn.Sequential(*layers)


    def _enable_cross_norm(self):
        active_cn_idxs = np.random.choice(self.cn_num, self.active_num, replace=False).tolist()
        assert len(set(active_cn_idxs)) == self.active_num
        
        #print(active_cn_idxs)
        
        for idx in active_cn_idxs:
            self.cn_modules[idx].active = True

        if self.block_idxs and 0 in self.block_idxs:
            self.img_cn.active = True

    def _disable_cross_norm(self):
        for i in range(len(self.cn_modules)):
            self.cn_modules[i].active = False
    def _forward_impl(self, x, aug=False):
        if self.block_idxs and 0 in self.block_idxs:
            x = self.img_cn(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        aux = x
        x = self.layer4(x)

        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.fc(x)

        return {'out': x, 'aux': aux}

    def forward(self, x, aug=False):


        return self._forward_impl(x, aug=aug)

def _resnet(arch, block, layers, pretrained, progress, SRM=False, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        if SRM:
            state_dict = torch.load('/research/cbim/vast/yg397/semseg/initmodel/Resnet_residual_1234_SRM.pth')
            print('model loaded from initmodel')
        else:
            state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict, strict=True)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlockCustom, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlockCustom, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, SRM=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', BottleneckCustom, [3, 4, 6, 3], pretrained, progress, SRM=SRM,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', BottleneckCustom, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', BottleneckCustom, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', BottleneckCustom, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', BottleneckCustom, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', BottleneckCustom, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', BottleneckCustom, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)



if __name__ == '__main__':
    import pdb 
    img = torch.randn((8, 3, 128, 128)).cuda()
    #net = resnet50(pretrained=True, pos='residual', cn_type='2ins_space', beta=1, block_idxs='0_1_2_3', crop='neither', sncn_type='srm', active_num=1)
    net = resnet50(pretrained=True, pos='post', beta=1, block_idxs='1_2_3_4', crop='neither', sncn_type='srm', active_num=1)

    net = net.cuda()
    #net = nn.DataParallel(net.cuda())
    out = net(img, aug=True)
    pdb.set_trace()
    print(out.shape)
