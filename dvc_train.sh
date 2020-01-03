#!/usr/bin/env bash

PY=/home/sergey/anaconda3/envs/tensorflow/bin/python

deps_code="-d consts.py -d train.py -d model.py -d image_data_generator.py"
deps_data="-d data/test_datset.pkl -d data/train_datset.pkl -d data/val_datset.pkl -d data/image_gen.pkl"

outputs="data/models"
metric="metric.txt"
dvc run $deps -f $outputs.dvc -o $outputs -M $metric $PY train.py
dvc push $outputs.dvc
