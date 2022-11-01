import fire
from sahi.scripts.slice_coco import slice
from tqdm import tqdm

MAX_WORKERS = 20
SLICE_SIZE_LIST = [300, 400, 500]
OVERLAP_RATIO_LIST = [0, 0.25]
IGNORE_NEGATIVE_SAMPLES = True


def slice_xview(image_dir: str, dataset_json_path: str, output_dir: str):
    total_run = len(SLICE_SIZE_LIST) * len(OVERLAP_RATIO_LIST)
    current_run = 1
    for slice_size in SLICE_SIZE_LIST:
        for overlap_ratio in OVERLAP_RATIO_LIST:
            tqdm.write(
                f"{current_run} of {total_run}: slicing for slice_size={slice_size}, overlap_ratio={overlap_ratio}"
            )
            slice(
                image_dir=image_dir,
                dataset_json_path=dataset_json_path,
                output_dir=output_dir,
                slice_size=slice_size,
                overlap_ratio=overlap_ratio,
            )
            current_run += 1


if __name__ == "__main__":
    # slice_xview(image_dir='/Users/mendeza/data/xview/train_images',
    #             dataset_json_path='/Users/mendeza/data/xview/train.json',
    #             output_dir='/Users/mendeza/data/xview/train_sliced/')
    slice_xview(image_dir='/Users/mendeza/data/xview/train_images',
                dataset_json_path='/Users/mendeza/data/xview/val.json',
                output_dir='/Users/mendeza/data/xview/val_sliced/')