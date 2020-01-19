
DATA_DIR='data'
RAW_DIR='raw'
MODELS_DIR='models'
MODELS_PRETRAINED_DIR='models_pretrained'
SUBMISSION_DIR = 'submissions'
SUBMISSION_CSV = 'submission.csv'
SAMPLE_SUBMISSION_CSV = 'sample_submission.csv'

METRIC_FILE_PATH='metric.txt'

TRAIN_IMAGE_DATA_PATTERN='train_image_data_*.parquet'
TEST_IMAGE_DATA_PATTERN='test_image_data_*.parquet'
TRAIN_CSV='train.csv'
TEST_CSV='test.csv'
CLASS_MAP_CSV='class_map.csv'

TRAIN_DATASET_PKL='train_datset.pkl'
VAL_DATASET_PKL='val_datset.pkl'
TEST_DATASET_PKL='test_datset.pkl'
IMAGE_GEN_PKL = 'image_gen.pkl'

MODEL_NAME='model'

IMG_WIDTH = 236
IMG_HEIGHT = 137
IMG_W=64
IMG_H=64
N_CHANNELS = 1
BATCH_SIZE=32
EPOCHS=10
LR=0.01

LR_SCHEDULER_PATINCE=0

SEED=0

TARGETS=['grapheme_root','vowel_diacritic','consonant_diacritic']

MODEL='model_pytorch'  #'model_tf'

try:
    import torch
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    print('Can not import pytorch and set cuda deterministic')

import numpy as np
np.random.seed(SEED)

import random
random.seed(SEED)
