#!/usr/bin/env bash

PY=/home/sergey/anaconda3/envs/tensorflow/bin/python

deps="-d make_submission.py -d data/models -d data/test_datset.pkl -d data/image_gen.pkl"
output_file_path=data/submissions/submission.csv
dvc run $deps -f $output_file_path.dvc -O $output_file_path $PY make_submission.py
