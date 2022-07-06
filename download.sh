#!/bin/bash

TO_PATH=./data/pretrained_model
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-airport-use-case/resnet101_caffe.pth -O $TO_PATH/resnet101_caffe.pth
TO_PATH=./data/cache
python3 -c "import os; os.makedirs('$TO_PATH', exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-airport-use-case/sate_airports_trainval_gt_roidb.pkl -O $TO_PATH/sate_airports_trainval_gt_roidb.pkl
TO_PATH=./data/cache
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-airport-use-case/sate_airports_trainval_sizes.pkl -O $TO_PATH/sate_airports_trainval_sizes.pkl
TO_PATH=./models/res101/sate_airports
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-airport-use-case/faster_rcnn_1_7_10021.pth -O $TO_PATH/faster_rcnn_1_7_10021.pth
TO_PATH=./dataset
python3 -c "import os; os.makedirs('$TO_PATH',exist_ok=True)"
wget https://image-quality-framework.s3-eu-west-1.amazonaws.com/iq-airport-use-case/SateAirports.zip -O $TO_PATH/SateAirports.zip
unzip $TO_PATH/SateAirports.zip -d $TO_PATH
