import os
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from model_base import ModelBase
from score import calc_score
from dataset_pytorch import BengaliDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
from typing import List
from torch.nn.parameter import Parameter
import math
from model import Model
from torch.nn import init
import pretrainedmodels
from torch.nn import Sequential
from cosine_scheduler import CosineAnnealingWarmUpRestarts
from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS,DATA_DIR,MODELS_DIR,MODEL_NAME
import matplotlib.pyplot as plt


from cosine_scheduler import CosineScheduler

classes=[167,10,7]

model_dir=os.path.join(DATA_DIR,MODELS_DIR)
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

model_filepath=os.path.join(model_dir,MODEL_NAME)

model_loaded=Model()
model_loaded.compile(classes)


optimizer=optim.Adam(model_loaded.parameters(),lr=LR)
iter_per_epochs=140000//BATCH_SIZE
scheduler = CosineScheduler(optimizer, period_initial=iter_per_epochs//2, period_mult=2, lr_initial=0.1, period_warmup_percent=0.1,lr_reduction=0.5)

lrs=[]
for epoch in range(EPOCHS):
    for i in range(iter_per_epochs):
        scheduler.step()
        lrs.append(scheduler.get_lr())

plt.plot(lrs)
plt.show()
