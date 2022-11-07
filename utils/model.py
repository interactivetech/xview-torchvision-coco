import sys
# sys.path.insert(0,'/run/determined/workdir')
import torch
import torchvision
# from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
from mobileone_fpn import mobileone
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor
from fcos import fcos_resnet50_fpn, FCOS
from torch import nn, Tensor
from typing import Callable, Dict, List, Optional, Union
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool

print("TORCHVISION_VERSION: ",torchvision.__version__, torchvision.__file__)
print("TORCH_VERSION: ",torch.__version__, torch.__file__)

class Backbone_FPN(nn.Module):
    def __init__(self,backbone: nn.Module,fpn: FeaturePyramidNetwork):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn

    def forward(self, x: Tensor)-> Dict[str, Tensor]:
        y = self.backbone(x)
        x = self.fpn(y)
        return x


def get_mobileone_s4_fpn_fcos(num_classes):
    backbone = mobileone(variant='s4', inference_mode=True)
    # ckpt = 'mobileone_s4.pth.tar'
    # checkpoint = torch.load(ckpt)
    # backbone.load_state_dict(checkpoint,strict=False)
    print("mobileone_s4.pth.tar loaded!")

    # fpn = FeaturePyramidNetwork([ 64,192, 448],256)
    # fpn = FeaturePyramidNetwork([ 64,192, 448,896],256)
    fpn = FeaturePyramidNetwork([ 64,192, 448,896,2048],256)


    b_fpn = Backbone_FPN(backbone,fpn)
    b_fpn.out_channels = 256
    # anchor_sizes = ( (16,), (32,), (64,))

    # anchor_sizes = ( (16,), (32,), (64,), (128,))
    anchor_sizes = ( (16,), (32,), (64,), (128,),(256,))

    anchor_generator = AnchorGenerator(
    sizes=anchor_sizes,
    aspect_ratios=((1.0,),)* len(anchor_sizes) 
    )   
    model = FCOS(
    b_fpn,
    num_classes=num_classes,
    anchor_generator=anchor_generator,
    )
    return model