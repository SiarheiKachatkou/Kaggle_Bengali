#!/usr/bin/env bash

set -x

for augm_count in {1000..10000..1000}
do

git_msg=mnist_augm_1_$augm_count
./dvc_create_dataset.sh $git_msg $augm_count
./dvc_train.sh $git_msg

git tag $git_msg

done
