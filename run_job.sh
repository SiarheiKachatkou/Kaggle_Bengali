#!/bin/bash

BUCKET_NAME=kochetkov_bucket_2
PROJECT_ID=core-memento-267713

DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=pytorch_bengali_$DATE
export GCS_JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
echo $GCS_JOB_DIR
export REGION=us-central1

gcloud config set project $PROJECT_ID

gcloud ai-platform jobs submit training $JOB_NAME \
    --config config.yaml \
    --runtime-version 1.15 \
    --stream-logs \
    --job-dir=$GCS_JOB_DIR \
    --package-path=trainer \
    --module-name trainer.task \
    --region $REGION -- \
    --ckpt_full_path= \
    --train_bin_files_dir=gs://$BUCKET_NAME/train_datset \
    --test_bin_files_dir=gs://$BUCKET_NAME/val_datset
