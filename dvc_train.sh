#!/usr/bin/env bash

set -x

git_cmt_message=$1

fine_tune=${2:-0}


if [[ $git_cmt_message == "" ]]
then
echo ERROR specify git commit message argument
exit 1
fi

PY=/home/sergey/anaconda3/envs/pytorch/bin/python

deps_code=""
for p in *.py
do
deps_code="$deps_code -d $p"
done


deps_data="-d data/test_datset.pkl -d data/train_datset.pkl -d data/val_datset.pkl"

outputs="data/models3"
metric="metric3.txt"

dvc run --overwrite-dvcfile $deps_code $deps_data -f $outputs.dvc -o $outputs -M $metric $PY train.py --fine_tune=$fine_tune &&
dvc push $outputs.dvc &&
git add $metric &&
git add $outputs.dvc -f &&
git commit -a -m $git_cmt_message
