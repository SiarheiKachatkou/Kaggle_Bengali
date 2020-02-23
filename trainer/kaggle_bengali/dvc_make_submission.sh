#!/usr/bin/env bash

PY=/home/sergey/anaconda3/envs/pytorch/bin/python

deps="-d make_submission.py -d data/models2 -d data/test_datset.pkl"
output_file_path=data/submissions/submission.csv
dvc run $deps -f $output_file_path.dvc -O $output_file_path $PY make_submission.py

git commit -a -m auto_cmt_submission
