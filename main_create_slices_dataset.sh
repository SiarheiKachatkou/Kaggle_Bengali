#!/usr/bin/env bash

set -x

TAG=467addd
DST_DIR=/home/sergey/1T/DNNDebug/Data/SlicesDataset/
CLASS_MAP_PATH='data/raw/class_map.csv'
MAX_IMGS_COUNT=10000
ACTIVATION_NAME=conv3
#_backbone.layer1.0.conv3


PY=/home/sergey/anaconda3/envs/pytorch/bin/python

for sub_dataset in train test
do
$PY main_create_slices_dataset.py --tag=$TAG --sub_dataset=$sub_dataset --dst_dir=$DST_DIR --max_imgs_count=$MAX_IMGS_COUNT --activation_name_postfix=$ACTIVATION_NAME --class_map_path=$CLASS_MAP_PATH
done
