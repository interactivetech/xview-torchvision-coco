import fiftyone as fo

name = "xview-dataset"
dataset_dir = "/Users/mendeza/data/xview/"
ann_path = '/Users/mendeza/data/xview/train.json'
# Create the dataset
dataset = fo.Dataset.from_dir(
    data_path=dataset_dir,
    labels_path=ann_path,
    dataset_type=fo.types.COCODetectionDataset,
    name=name,
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())