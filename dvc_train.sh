#!/usr/bin/env bash

git_cmt_message=$1

set -x

PY=/home/sergey/anaconda3/envs/tensorflow/bin/python

deps_code=""
for p in *.py
do
deps_code=$deps_code -d $p
done


deps_data="-d data/test_datset.pkl -d data/train_datset.pkl -d data/val_datset.pkl -d data/image_gen.pkl"

outputs="data/models"
metric="metric.txt"
dvc run $deps_code $deps_data -f $outputs.dvc -o $outputs -M $metric $PY train.py
dvc push $outputs.dvc

git commit -a -m $git_cmt_message
