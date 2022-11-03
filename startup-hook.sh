nvidia-smi
apt-get update
apt-get install unzip

# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/train_sliced_no_neg/train_300_02.json
wget https://determined-ai-xview-coco-dataset.s3.us-west-2.amazonaws.com/val_sliced_no_neg/val_300_02.json
mv train_300_02.json /tmp
mv val_300_02.json/tmp

# pip install determined
# pip install torch --upgrade
# pip install torchvision --upgrade
# pip3 install torch torchvision torchaudio --upgrade --extra-index-url https://download.pytorch.org/whl/cu113
# pip install torch==1.11.+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch torchvision torchaudio --upgrade --extra-index-url https://download.pytorch.org/whl/cu113

# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install pycocotools

mkdir ~/.aws/
cp credentials ~/.aws/
wget https://docs-assets.developer.apple.com/ml-research/datasets/mobileone/mobileone_s4.pth.tar
# export PYTHONPATH=/run/determined/workdir/vision:$PYTHONPATH
# echo $PYTHONPATH