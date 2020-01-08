#!/usr/bin/env bash

set -x

TAG=7ab610
DST_DIR=/home/sergey/1T/DNNDebug/Data/SlicesDataset/
MAX_IMG_COUNT=1000

PY=/home/sergey/anaconda3/envs/pytorch/bin/python

for sub_dataset in train test
do
$PY main_create_slices_dataset.py --tag=$TAG --sub_dataset=$sub_dataset --dst_dir=$DST_DIR --max_img_count=$MAX_IMG_COUNT
done
