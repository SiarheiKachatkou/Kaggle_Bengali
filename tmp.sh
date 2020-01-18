#!/usr/bin/env bash


set -x

MODE=BENGALI

git_commit_message=$1

augm_count=$2

if [[ $git_commit_message == "" ]]
then
echo ERROR specify git commit message ar

deps_code=""
for p in *.py
do
deps_code="$deps_code -d $p"
done

deps_data="-d data/raw"

outputs="-o data/test_datset.pkl -o data/train_datset.pkl -o data/val_datset.pkl"
