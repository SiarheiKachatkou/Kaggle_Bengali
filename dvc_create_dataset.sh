#!/usr/bin/env bash

set -x

MODE=BENGALI

# MNIST


git_commit_message=$1
augm_count=$2

if [[ $git_commit_message == "" ]]
then
echo ERROR specify git commit message argument
exit 1
fi

PY=/home/user/anaconda3/envs/pytorch/bin/python

deps_code=""
for p in *.py
do
deps_code="$deps_code -d $p"
done

deps_data="-d data/raw"

outputs="-o data/test_datset -o data/train_datset -o data/val_datset"

if [[ $MODE == MNIST ]]
then
py_script="create_dataset_mnist.py --augm_count=$augm_count"
else
py_script=create_dataset.py
fi

output_dvc=data/train_val_test_datset.pkl.dvc
dvc run --overwrite-dvcfile $deps_data $deps_code -f $output_dvc $outputs $PY $py_script &&
dvc push $output_dvc &&
git add $output_dvc -f &&
git commit -a -m $git_commit_message
