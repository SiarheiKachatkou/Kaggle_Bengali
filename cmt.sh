#!/bin/bash

cmt_msg=$1

git commit -a -m $cmt_msg
git push
