# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# unzip annotations_trainval2017.zip 
# mv annotations/instances_train2017.json /tmp
# mv annotations/instances_val2017.json /tmp

# wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg/train_300_02.json"
# mv train_300_02.json /tmp
# wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg/val_300_02.json"
# mv val_300_02.json /tmp

wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg.tar.gz
mv train_sliced_no_neg.tar.gz /tmp
tar -xvf /tmp/train_sliced_no_neg.tar.gz -C /tmp

wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg.tar.gz
mv val_sliced_no_neg.tar.gz /tmp
tar -xvf /tmp/val_sliced_no_neg.tar.gz -C /tmp