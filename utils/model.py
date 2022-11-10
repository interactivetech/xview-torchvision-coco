import sys
# sys.path.insert(0,'/run/determined/workdir')
import torch
import torchvision
from torchvision.models.detection import FCOS
from torchvision.models.detection.anchor_utils import AnchorGenerator
from utils.mobileone_fpn import mobileone
from torchvision.models.detection.backbone_utils import _mobilenet_extractor, _resnet_fpn_extractor
from torch import nn, Tensor
from typing import Callable, Dict, List, Optional, Union
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
import torchsummary

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

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


def get_mobileone_s4_fpn_fcos(num_classes, ckpt_path=None):
    backbone = mobileone(variant='s4', inference_mode=True)
    # ckpt = 'mobileone_s4.pth.tar'
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path)
        backbone.load_state_dict(checkpoint,strict=False)
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

def make_custom_object_detection_model_fcos(num_classes):
    model = fcos_resnet50_fpn(pretrained=True)  # load an object detection model pre-trained on COCO
    model.score_thresh = 0.05
    model.nms_thresh = 0.4
    model.detections_per_img = 300
    model.topk_candidates = 300
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head.num_classes = num_classes
    print("FOCS num_classes: ",model.head.classification_head.num_classes)

    out_channels = model.head.classification_head.conv[9].out_channels
    cls_logits = torch.nn.Conv2d(out_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
    torch.nn.init.normal_(cls_logits.weight, std=0.01)
    torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    model.head.classification_head.cls_logits = cls_logits

def build_frcnn_model(num_classes):
    # load an detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.min_size=800
    model.max_size=1333
    # RPN parameters
    model.rpn_pre_nms_top_n_train=2000
    model.rpn_pre_nms_top_n_test=1000
    model.rpn_post_nms_top_n_train=2000
    model.rpn_post_nms_top_n_test=1000
    model.rpn_nms_thresh=0.7
    model.rpn_fg_iou_thresh=0.7
    model.rpn_bg_iou_thresh=0.3
    model.rpn_batch_size_per_image=256
    model.rpn_positive_fraction=0.5
    model.rpn_score_thresh=0.0
    # Box parameters
    model.box_score_thresh=0.05
    model.box_nms_thresh=0.5
    model.box_detections_per_img=100
    model.box_fg_iou_thresh=0.5
    model.box_bg_iou_thresh=0.5
    model.box_batch_size_per_image=512
    model.box_positive_fraction=0.25
    return model

if __name__ == '__main__':
    model = get_mobileone_s4_fpn_fcos(91,ckpt_path='/tmp/mobileone_s4.pth.tar')
    # model = build_frcnn_model(61)
    # model.eval()
    torchsummary.summary(model,input_size=(3,256,256),device='cpu')