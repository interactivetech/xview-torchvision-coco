import torch
from utils.data import build_dataset,build_xview_dataset, unwrap_collate_fn
from attrdict import AttrDict
from utils.group_by_aspect_ratio import create_aspect_ratio_groups, GroupedBatchSampler
from torchvision.models.detection import fcos_resnet50_fpn
import time
import datetime
from tqdm import tqdm
from progress.bar import Bar

import torchvision
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from utils.coco_eval import CocoEvaluator
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
    #Data loading code
    device = torch.device('cpu')
    cpu_device = torch.device('cpu')
    print("Loading data")
    
    # dataset, num_classes = data = build_dataset(image_set="train", args=AttrDict({
    #                                             'data_dir':'determined-ai-coco-dataset',
    #                                             'backend':'aws',
    #                                             'masks': False,
    #                                             }))
    # dataset_test, _ = build_dataset(image_set="val", args=AttrDict({
    #                                             'data_dir':'determined-ai-coco-dataset',
    #                                             'backend':'aws',
    #                                             'masks': False,
    #                                             }))
    # TRAIN_DATA_DIR='/tmp/train_sliced_no_neg/train_images_300_02/'
    # VAL_DATA_DIR='/tmp/val_sliced_no_neg/val_images_300_02/'
    TRAIN_DATA_DIR='/run/determined/workdir/xview-torchvision-coco/train_sliced_no_neg/train_images_300_02/'
    VAL_DATA_DIR='/run/determined/workdir/xview-torchvision-coco/val_sliced_no_neg/val_images_300_02/'
    dataset, num_classes =  build_xview_dataset(image_set='train',args=AttrDict({
                                                'data_dir':TRAIN_DATA_DIR,
                                                'backend':'local',
                                                'masks': None,
                                                }))
    dataset_test, _ = build_xview_dataset(image_set='val',args=AttrDict({
                                                'data_dir':VAL_DATA_DIR,
                                                'backend':'local',
                                                'masks': None,
                                                }))
#     TRAIN_DATA_DIR='determined-ai-xview-coco-dataset/train_sliced_no_neg/train_images_300_02/'

#     dataset, n_classes = build_xview_dataset(image_set='train',args=AttrDict({
#                                                 'data_dir':TRAIN_DATA_DIR,
#                                                 'backend':'aws',
#                                                 'masks': None,
#                                                 }))
#     VAL_DATA_DIR='determined-ai-xview-coco-dataset/val_sliced_no_neg/val_images_300_02/'
#     dataset_test, n_classes = build_xview_dataset(image_set='val',args=AttrDict({
#                                                 'data_dir':VAL_DATA_DIR,
#                                                 'backend':'aws',
#                                                 'masks': None,
#                                                 }))
    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    group_ids = create_aspect_ratio_groups(dataset, k=3)
    train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, batch_size=16)

    train_collate_fn = unwrap_collate_fn
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_sampler=train_batch_sampler, num_workers=2, collate_fn=train_collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, sampler=test_sampler, num_workers=2, collate_fn=train_collate_fn)
    
    print("Create Model")
    model = fcos_resnet50_fpn(pretrained=False,num_classes=91)
    model.to(device)
    parameters = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.SGD(
            parameters,
            lr=0.01,
            momentum=0.9,
            weight_decay=1e-4,
            nesterov="nesterov",
        )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16,22], gamma=0.1)

    print("Start training")
    start_time = time.time()
    model.train()
    for e in range(1):
        # bar = Bar('Processing', max=len(data_loader))
        it=0
        pbar = tqdm(enumerate(data_loader),total=len(data_loader))
        # Train Batch
        for ind, (images, targets) in pbar:
            optimizer.zero_grad()
            batch_time_start = time.time()
            images = list(image.to(device,non_blocking=True) for image in images)
            targets = [{k: v.to(device,non_blocking=True) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses_reduced = sum(loss for loss in loss_dict.values())
            loss_value = losses_reduced.item()
            # if ind %10 == 0:
            # print("loss: ",loss_value)
            # print("losses_reduced: ",losses_reduced)
            losses_reduced.backward()
            optimizer.step()
            total_batch_time = time.time() - batch_time_start
            total_batch_time_str = str(datetime.timedelta(seconds=int(total_batch_time)))
            # print(f"Training time {total_batch_time_str}")
            # bar.suffix = ("Iter: {batch:4}/{iter:4}: {loss:4}.".format(batch=it, iter=len(data_loader),loss=loss_value))
            # bar.next()
            it += 1
            pbar.set_postfix({'loss': loss_value})
        # bar.finish()
            # break
        lr_scheduler.step()

        # Eval
        coco = get_coco_api_from_dataset(data_loader.dataset)
        iou_types = ['bbox']
        coco_evaluator = CocoEvaluator(coco, iou_types)
        model.eval()
        for ind, (images, targets) in tqdm(enumerate(data_loader_test),total=len(data_loader_test)):
            with torch.no_grad():
                model_time = time.time()
                outputs = model(images)

                outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
                model_time = time.time() - model_time
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                model_time_str = str(datetime.timedelta(seconds=int(model_time)))
                print("Model Time: ",model_time_str)

                evaluator_time = time.time()
                coco_evaluator.update(res)
                evaluator_time = time.time() - evaluator_time
                evaluator_time_str = str(datetime.timedelta(seconds=int(evaluator_time)))
                print("COCO Eval Time: ",evaluator_time_str)

        # accumulate predictions from all images
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

if __name__ == '__main__':
    main()