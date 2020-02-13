import logging
import os

ARTIFACTS_DIR='artifacts'
DATA_DIR='data'
RAW_DIR='raw'
MODELS_DIR='models'
MODELS_PRETRAINED_DIR='models_pretrained'
SUBMISSION_DIR = 'submissions'
SUBMISSION_CSV = 'submission.csv'
SAMPLE_SUBMISSION_CSV = 'sample_submission.csv'

LOG_FILENAME=os.path.join(os.path.dirname(__file__),'training_log.txt')

LOG_LEVEL=logging.INFO

METRIC_FILE_PATH='metric.txt'

TRAIN_IMAGE_DATA_PATTERN='train_image_data_*.parquet'
TEST_IMAGE_DATA_PATTERN='test_image_data_*.parquet'
TRAIN_CSV='train.csv'
TEST_CSV='test.csv'
CLASS_MAP_CSV='class_map.csv'

TRAIN_DATASET_DIR='train_datset'
VAL_DATASET_DIR='val_datset'
TEST_DATASET_DIR='test_datset'
IMAGE_GEN_PKL = 'image_gen.pkl'

MODEL_NAME='model'

IMG_WIDTH = 236
IMG_HEIGHT = 137
IMG_W=64
IMG_H=64
TOP_CUT=4
LEFT_CUT=4
PAD=4


N_CHANNELS = 1
BATCH_SIZE=1024
EPOCHS=100
LR=0.05
LR_SCHEDULER_PATINCE=12
AUGM_PROB=0.4
DROPOUT_P=0.5

LOSS_WEIGHTS=[1,1,1]
BETA=0

USE_FREQ_SAMPLING=False

FAST_PROTO_SCALE=1

CLASSES_LIST=[168,11,7]

SEED=0

TARGETS=['grapheme_root','vowel_diacritic','consonant_diacritic']

try:
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    print('can not import pytorch, gpu use will be undeterministic')

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)
