# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# unzip annotations_trainval2017.zip 
# mv annotations/instances_train2017.json /tmp
# mv annotations/instances_val2017.json /tmp
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102
pip install boto pycocotools attrdict progress

wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg/train_300_02.json"
mkdir /tmp/train_sliced_no_neg/
mv train_300_02.json /tmp/train_sliced_no_neg/train_300_02.json 
wget "https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg/val_300_02.json"
mkdir /tmp/val_sliced_no_neg
mv val_300_02.json /tmp/val_sliced_no_neg/val_300_02.json

# wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg.tar.gz
# mv train_sliced_no_neg.tar.gz /tmp
# tar -xvf /tmp/train_sliced_no_neg.tar.gz  -C /tmp
# tar -xvf train_sliced_no_neg.tar.gz 

# wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg.tar.gz
# mv val_sliced_no_neg.tar.gz /tmp
# tar -xvf /tmp/val_sliced_no_neg.tar.gz -C /tmp
