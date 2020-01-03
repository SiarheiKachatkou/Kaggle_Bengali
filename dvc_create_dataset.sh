#!/usr/bin/env bash

PY=/home/sergey/anaconda3/envs/tensorflow/bin/python

deps="-d consts.py -d create_dataset.py -d image_data_generator.py"
outputs="-o data/test_datset.pkl -o data/train_datset.pkl -o data/val_datset.pkl -o data/image_gen.pkl"

output_dvc=data/train_val_test_datset.pkl.dvc
dvc run $deps -f $output_dvc $outputs $PY create_dataset.py
dvc push $output_dvc
