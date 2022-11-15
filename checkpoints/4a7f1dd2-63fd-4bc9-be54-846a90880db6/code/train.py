import torch
from utils.data import build_dataset,build_xview_dataset, unwrap_collate_fn
from attrdict import AttrDict
from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from utils.fcos import fcos_resnet50_fpn
# from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import ssd300_vgg16
import time
import datetime
from tqdm import tqdm

import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from utils.coco_eval import CocoEvaluator
import math
from lr_schedulers import WarmupWrapper
from torch.optim.lr_scheduler import MultiStepLR

from utils.model import make_custom_object_detection_model_fcos, build_frcnn_model

def collate_fn(batch):
    return tuple(list(zip(*batch)))

def convert_to_coco_api(ds):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # make masks Fortran contiguous for coco_mask
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id
            if "masks" in targets:
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds
def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)

def main():
    DEVICE='cuda'
    #Data loading code
    device = torch.device(DEVICE)
    cpu_device = torch.device(DEVICE)
    print("Loading data")
    
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
    TRAIN_DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'

    dataset, num_classes = build_xview_dataset(image_set='train',args=AttrDict({
                                                'data_dir':TRAIN_DATA_DIR,
                                                'backend':'aws',
                                                'masks': None,
                                                }))
    print("--num_classes: ",num_classes)
    VAL_DATA_DIR='determined-ai-xview-coco-dataset/val_sliced_no_neg/val_images_300_02/'
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
        dataset, batch_size=8,batch_sampler=None,shuffle=True, num_workers=2, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, sampler=test_sampler, num_workers=2, collate_fn=train_collate_fn)
    
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
    
    losses = []
    it=0
    for e in range(20):
        
        pbar = tqdm(enumerate(data_loader),total=len(data_loader))
        # Train Batch
        model.train()
        for ind, (images, targets) in pbar:
            batch_time_start = time.time()
            images = list(image.to(device,non_blocking=True) for image in images)
            targets = [{k: v.to(device,non_blocking=True) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses_reduced = sum(loss for loss in loss_dict.values())
            
            # if ind %10 == 0:
            # print("loss: ",loss_value)
            # print("losses_reduced: ",losses_reduced)
            optimizer.zero_grad()
            losses_reduced.backward()
            optimizer.step()
            with torch.no_grad():
                # print(losses_reduced)
                loss_value = losses_reduced.item()
            total_batch_time = time.time() - batch_time_start
            total_batch_time_str = str(datetime.timedelta(seconds=int(total_batch_time)))
            # print(f"Training time {total_batch_time_str}")
            if it%10==0:
                losses.append(loss_value)
            it += 1
            loss_str = []
            loss_str.append("{}: {}".format("loss","{:.3f}".format(loss_value)))
            for name, val in loss_dict.items():
                loss_str.append(
                    "{}: {}".format(name, "{:.3f}".format(val) )
                )
                pbar.set_postfix({'loss': loss_str})
            print(it, scheduler.get_last_lr())
            scheduler.step()
            # if ind>100:
            # break
            # break
            # break
        # lr_scheduler.step()

            # Eval
        coco = get_coco_api_from_dataset(data_loader_test.dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(coco, iou_types)
        torch.save(model,'model.pth')
        model.eval()
        pbar = tqdm(enumerate(data_loader_test),total=len(data_loader_test))
        for ind, (images, targets) in pbar:
            with torch.no_grad():
                model_time = time.time()
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                # loss_dict, outputs = model(images,targets)
                outputs = model(images)
                losses_reduced = sum(loss for loss in loss_dict.values())
                # with torch.no_grad():
                    # print("loss_dict: ",loss_dict)
                    # print(" losses_reduced val: ",losses_reduced)
                loss_value = losses_reduced.item()
                # outputs = model(images)

                # print(type(outputs[0]))
                # print(outputs)
#                 loss_str = []
#                 loss_str.append("{}: {}".format("loss","{:.3f}".format(loss_value)))

#                 for name, val in loss_dict.items():
#                     loss_str.append(
#                         "{}: {}".format(name, "{:.3f}".format(val) )
#                     )
#                     pbar.set_postfix({'loss': loss_str})
                outputss = []
                for t in outputs:
                    outputss.append({k: v.to(cpu_device) for k, v in t.items()})
                model_time = time.time() - model_time
                res = {target["image_id"].item(): output for target, output in zip(targets, outputss)}
                model_time_str = str(datetime.timedelta(seconds=int(model_time)))
                # print("Model Time: ",model_time_str)

                evaluator_time = time.time()
                # print("data_loader_test.dataset.catIdtoCls: ",data_loader_test.dataset.clstoCatId)
                coco_evaluator.update(res,remap_dict=data_loader_test.dataset.clstoCatId)
                evaluator_time = time.time() - evaluator_time
                evaluator_time_str = str(datetime.timedelta(seconds=int(evaluator_time)))
                # print("COCO Eval Time: ",evaluator_time_str)
        
        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    import matplotlib.pyplot as plt
    plt.plot(range(len(losses)),losses )
    plt.title("Loss")
    plt.legend(['loss'])
    plt.savefig("loss.png")

if __name__ == '__main__':
    main()