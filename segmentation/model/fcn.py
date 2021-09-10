from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import model.resnet as models
import model.cnsn_resnet as cnsn_resnet
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101
from .utils import PyramidPooling
import time

import pdb
class FCNet(nn.Module):

    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True):
        
        super(FCNet, self).__init__()
        assert layers in [50, 101]
        assert classes > 1
        self.criterion = criterion


        if layers == 50:
            self.model = fcn_resnet50(pretrained=False, progress=True, num_classes=classes, aux_loss=True)
            resnet = models.resnet50(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
        elif layers == 101:
            self.model = fcn_resnet101(pretrained=False, progress=True, num_classes=classes, aux_loss=True)
            resnet = models.resnet101(pretrained=pretrained, replace_stride_with_dilation=[False, True, True])
        
        return_layers = {'layer4': 'out'}
        return_layers['layer3'] = 'aux'
        
        backbone = IntermediateLayerGetter(resnet, return_layers=return_layers)

        self.model.backbone = backbone

    
    def forward(self, x, y=None):
        
        result = self.model(x)
        x = result['out']
        aux = result['aux']

        if self.training:
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)

            return x.max(1)[1], main_loss, aux_loss

        else:
            return x

class FCN_RESNET(_SimpleSegmentationModel):
    
    def forward(self, x, aug=False):
        
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result["aux"] = x

        return result, features
    
         



class FCN_CNSN(nn.Module):

    def __init__(self, layers=50, classes=2, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, config=None):
        
        super(FCN_CNSN, self).__init__()
        assert layers in [50] # only support FCN-50 right now
        assert classes > 1
        self.criterion = criterion
        self.config = config
        self.l1_loss = nn.L1Loss()
        
        if 'sn' in config.cnsn_type:
            SN = True
        else:
            SN = False

        resnet = cnsn_resnet.resnet50(pretrained=pretrained, SN=SN, replace_stride_with_dilation=[False, True, True], pos=config.pos, cn_pos=config.cn_pos, beta=config.beta, block_idxs=config.block_idxs, crop=config.crop, cnsn_type=config.cnsn_type, active_num=config.active_num)

       
        classifier = FCNHead(2048, classes)
        aux_classifier = FCNHead(1024, classes)

        self.model = FCN_RESNET(resnet, classifier, aux_classifier) 
        

    def forward(self, x, y=None):
        img = x

        result, features = self.model(x, aug=aug)
        x = result['out']
        aux = result['aux']
        

        feat_main = features['out']
        feat_aux = features['aux']

        if self.training:

            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            
            return x.max(1)[1], main_loss, aux_loss

        else:
            return x


