#!/usr/bin/env bash

set -x

MODE=MNIST
# BENGALI

git_commit_message=$1

if [[ $git_commit_message == "" ]]
then
echo ERROR specify git commit message argument
exit 1
fi

PY=/home/sergey/anaconda3/envs/tensorflow/bin/python

deps_code=""
for p in *.py
do
deps_code="$deps_code -d $p"
done

deps_data="-d data/raw"

outputs="-o data/test_datset.pkl -o data/train_datset.pkl -o data/val_datset.pkl"

if [[ $MODE == MNIST ]]
then
py_script=create_dataset_mnist.py
else
py_script=create_dataset.py
fi

output_dvc=data/train_val_test_datset.pkl.dvc
dvc run --overwrite-dvcfile $deps_data $deps_code -f $output_dvc $outputs $PY $py_script
dvc push $output_dvc
git commit -a -m $git_commit_message
