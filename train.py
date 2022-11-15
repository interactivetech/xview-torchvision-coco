import torch
from utils.data import build_dataset,build_xview_dataset, unwrap_collate_fn
from attrdict import AttrDict
from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from utils.fcos import fcos_resnet50_fpn
# from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import ssd300_vgg16

import datetime
import time
from tqdm import tqdm

from utils.engine import train_and_eval,eval_model
import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
import math
from lr_schedulers import WarmupWrapper
from torch.optim.lr_scheduler import MultiStepLR

from utils.model import make_custom_object_detection_model_fcos, build_frcnn_model
import matplotlib.pyplot as plt

def collate_fn(batch):
    return tuple(list(zip(*batch)))



def load_dataset(TRAIN_DATA_DIR=None,VAL_DATA_DIR=None,train_batch_size=None,test_batch_size=None):
    '''
    '''
     # dataset, num_classes = data = build_dataset(image_set="train", args=AttrDict({
    #                                             'data_dir':'determined-ai-coco-dataset',
    #                                             'backend':'aws1',
    #                                             'masks': False,
    #                                             }))
    # dataset_test, _ = build_dataset(image_set="val", args=AttrDict({
    #                                             'data_dir':'determined-ai-coco-dataset',
    #                                             'backend':'aws1',
    #                                             'masks': False,
    #                                             }))
    # TRAIN_DATA_DIR='/tmp/train_sliced_no_neg/train_images_300_02/'
    # VAL_DATA_DIR='/tmp/val_sliced_no_neg/val_images_300_02/'
    # # TRAIN_DATA_DIR='/run/determined/workdir/xview-torchvision-coco/train_sliced_no_neg/train_images_300_02/'
    # # VAL_DATA_DIR='/run/determined/workdir/xview-torchvision-coco/val_sliced_no_neg/val_images_300_02/'
    # dataset, num_classes =  build_xview_dataset(image_set='train',args=AttrDict({
    #                                             'data_dir':TRAIN_DATA_DIR,
    #                                             'backend':'local',
    #                                             'masks': None,
    #                                             }))
    # print("NUM Classes: ",num_classes)
    # dataset_test, _ = build_xview_dataset(image_set='val',args=AttrDict({
    #                                             'data_dir':VAL_DATA_DIR,
    #                                             'backend':'local',
    #                                             'masks': None,
    #                                             }))

    dataset, num_classes = build_xview_dataset(image_set='train',args=AttrDict({
                                                'data_dir':TRAIN_DATA_DIR,
                                                'backend':'aws',
                                                'masks': None,
                                                }))
    print("--num_classes: ",num_classes)
    dataset_test, _ = build_xview_dataset(image_set='val',args=AttrDict({
                                                'data_dir':VAL_DATA_DIR,
                                                'backend':'aws',
                                                'masks': None,
                                               }))
    # FOR DEBUG PURPOSES
    # VAL_DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'
    # dataset_test, _ = build_xview_dataset(image_set='val',args=AttrDict({
    #                                             'data_dir':VAL_DATA_DIR,
    #                                             'backend':'aws',
    #                                             'masks': None,
    #                                             }))
    print("Creating data loaders")
    # train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    # group_ids = create_aspect_ratio_groups(dataset, k=3)
    # train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size=16)

    train_collate_fn = unwrap_collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size,batch_sampler=None,shuffle=True, num_workers=2, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size, sampler=test_sampler, num_workers=2, collate_fn=train_collate_fn)
    return dataset, num_classes, dataset_test,data_loader, data_loader_test

def main():
    DEVICE='cuda'
    #Data loading code
    device = torch.device(DEVICE)
    cpu_device = torch.device(DEVICE)
    print("Loading data")
    TRAIN_DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'
    VAL_DATA_DIR='determined-ai-xview-coco-dataset/val_sliced_no_neg/val_images_300_02/'

    dataset, num_classes, dataset_test,data_loader, data_loader_test= load_dataset(TRAIN_DATA_DIR=TRAIN_DATA_DIR,VAL_DATA_DIR=VAL_DATA_DIR,train_batch_size=8,test_batch_size=8)
    print("Create Model")
    # model = fcos_resnet50_fpn(pretrained=False,num_classes=num_classes+1)
    # model = make_custom_object_detection_model_fcos(dataset.num_classes)
    model = build_frcnn_model(dataset.num_classes)
    # model = ssd300_vgg16(pretrained=False,num_classes=91)
    model.to(device)
    # parameters = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov="nesterov",
        )
    # optimizer = torch.optim.SGD(
    #         parameters,
    #         lr=1e-4,
    #         momentum=0.9,
    #         weight_decay=1e-4,
    #         nesterov="nesterov",
    #     )
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16,22], gamma=0.1)
    scheduler_cls = WarmupWrapper(MultiStepLR)
    scheduler = scheduler_cls(
        'linear',  # warmup schedule
        100,  # warmup_iters
        0.001,  # warmup_ratio
        optimizer,
        [177429, 236572],  # milestones
        0.1,  # gamma
    )
    print("Start training")
    start_time = time.time()
    
    losses, model = train_and_eval(model,data_loader,data_loader_test,optimizer,scheduler,device,cpu_device,epochs=1)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    plt.plot(range(len(losses)),losses )
    plt.title("Loss")
    plt.legend(['loss'])
    plt.savefig("loss.png")

if __name__ == '__main__':
    main()