#!/usr/bin/env bash

set -x

git_cmt_message=$1

fine_tune=${2:-0}


if [[ $git_cmt_message == "" ]]
then
echo ERROR specify git commit message argument
exit 1
fi

source export_env.sh

deps_code=""
for p in *.py
do
deps_code="$deps_code -d $p"
done


deps_data="-d data/test_datset -d data/train_datset -d data/val_datset"

outputs="data/models"
metric="metric.txt"

train_bin_files_dir=/home/sergey/1T/Bengali_github/data/train_datset
test_bin_files_dir=/home/sergey/1T/Bengali_github/data/val_datset


dvc run --overwrite-dvcfile $deps_code $deps_data -f $outputs.dvc -o $outputs -M $metric $PY run_train.py --train_bin_files_dir=$train_bin_files_dir --test_bin_files_dir=$test_bin_files_dir &&
dvc push $outputs.dvc &&
git add $metric &&
git add $outputs.dvc -f &&
git commit -a -m $git_cmt_message
