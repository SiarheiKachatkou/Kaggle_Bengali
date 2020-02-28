import logging
import os
import numpy as np
import random
from .resnet import Bottleneck,SEBottleneck,_resnet
from .inceptionresnetv2 import InceptionResNetV2
from .inceptionv4 import InceptionV4

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

alpha=1.2
beta=1.1
gama=1.15
phi=2


IMG_WIDTH = 236
IMG_HEIGHT = 137
IMG_W=int(64*gama**phi)
IMG_H=int(64*gama**phi)
TOP_CUT=4
LEFT_CUT=4
PAD=4


N_CHANNELS = 1
BATCH_SIZE=64
EPOCHS=10
LR=0.001
LR_SCHEDULER_PATINCE=8000
AUGM_PROB=0
DROPOUT_P=0.5

LOSS_WEIGHTS=[1,1,1]
BETA=0

CLASSES_LIST=[168,11,7]


_m=alpha**phi


RESNEXT_KWARGS={'arch':'small_resnet', 'groups': 32,'width_per_group': int(8*beta**phi), 'block':Bottleneck, 'layers':[int(_m*2), int(_m*2), int(_m*2), int(_m*2)], 'num_classes':np.sum(CLASSES_LIST),'pretrained':False, 'progress':False}

RESNET_KWARGS={'arch':'small_resnet', 'width_per_group': int(64*beta**phi), 'block':Bottleneck, 'layers':[int(_m*2), int(_m*2), int(_m*2), int(_m*2)], 'num_classes':np.sum(CLASSES_LIST),'pretrained':False, 'progress':False}

SERESNET_KWARGS={'arch':'small_resnet', 'width_per_group': int(64*beta**phi), 'block':SEBottleneck, 'layers':[int(_m*2), int(_m*2), int(_m*2), int(_m*2)], 'num_classes':np.sum(CLASSES_LIST),'pretrained':False, 'progress':False}

SERESNEXT_KWARGS={'arch':'small_resnet', 'groups': 32,'width_per_group': int(8*beta**phi), 'block':SEBottleneck, 'layers':[int(_m*2), int(_m*2), int(_m*2), int(_m*2)], 'num_classes':np.sum(CLASSES_LIST),'pretrained':False, 'progress':False}

INCEPTIONRESNETV2_KWARGS={'repeats':(int(_m*3),int(_m*6),int(_m*4)), 'width':0.3*beta**phi, 'num_classes':np.sum(CLASSES_LIST)}

INCEPTIONV4_KWARGS={'repeats':(int(_m*2),int(_m*4),int(_m*2)), 'width':0.3*beta**phi, 'num_classes':np.sum(CLASSES_LIST)}

BACKBONE_KWARGS=RESNET_KWARGS
BACKBONE_FN=_resnet


TARGETS=['grapheme_root','vowel_diacritic','consonant_diacritic']

SEED=0

import torch
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

np.random.seed(SEED)
random.seed(SEED)
