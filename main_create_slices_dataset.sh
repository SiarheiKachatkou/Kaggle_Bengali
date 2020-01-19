#!/usr/bin/env bash

set -x

TAG=d08460b
DST_DIR=/home/user/GIT/DNNDebug/Data/SlicesDataset/
CLASS_MAP_PATH='data/raw/class_map.csv'
MAX_IMGS_COUNT_TRAIN=10000
MAX_IMGS_COUNT_TEST=100
ACTIVATION_NAME=conv3
#_backbone.layer1.0.conv3


PY=/home/user/anaconda3/envs/pytorch/bin/python

sub_datasets=(train test)
max_sample_counts=($MAX_IMGS_COUNT_TRAIN $MAX_IMGS_COUNT_TEST)

for i in 0 1
do
$PY main_create_slices_dataset.py --tag=$TAG --sub_dataset=${sub_datasets[$i]} --dst_dir=$DST_DIR --max_imgs_count=${max_sample_counts[$i]} --activation_name_postfix=$ACTIVATION_NAME --class_map_path=$CLASS_MAP_PATH
done
