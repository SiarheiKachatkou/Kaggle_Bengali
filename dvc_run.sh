#!/usr/bin/env bash

set -x

for augm_count in 800 900 1000 2000 4000 8000
do

git_msg=mnist_augm_$augm_count
./dvc_create_dataset.sh $git_msg $augm_count
./dvc_train.sh $git_msg

git tag $git_msg

done
