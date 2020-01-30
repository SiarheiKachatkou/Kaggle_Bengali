#!/usr/bin/env bash

PY=/home/user/anaconda3/envs/pytorch/bin/python
if ! [[ -f $PY ]]
then
PY=/home/sergey/anaconda3/envs/pytorch/bin/python

if ! [[ -f $PY ]]
then
PY=python3
fi

fi
