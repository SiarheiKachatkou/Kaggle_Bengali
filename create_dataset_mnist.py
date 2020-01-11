import pandas as pd
import os
import glob
import cv2
import torch
import torchvision
from torchvision import transforms
import numpy as np
import sklearn.model_selection
from create_dataset_utils import dump,load

from consts import BATCH_SIZE, DATA_DIR,RAW_DIR,TRAIN_IMAGE_DATA_PATTERN, TEST_IMAGE_DATA_PATTERN, IMG_HEIGHT,IMG_WIDTH,N_CHANNELS,TRAIN_CSV,CLASS_MAP_CSV, IMG_H,IMG_W, TRAIN_DATASET_PKL, VAL_DATASET_PKL, TEST_DATASET_PKL, IMAGE_GEN_PKL, SEED, TARGETS

debug_mode=False

def preproc(x):
    x=cv2.resize(x,(IMG_W,IMG_H))
    return np.expand_dims(x,axis=-1)

def data_loader_to_array(data_loader):
    imgs=[]
    labels=[]
    ids=[]
    img_idx=0
    for batch_img,batch_label in data_loader:

        imgs.extend([preproc(img) for img in batch_img.cpu().numpy()])
        labels.append(batch_label.cpu().numpy())
        for _ in range(len(batch_img)):
            ids.extend(img_idx)
            img_idx+=1

    imgs=np.concatenate(imgs,axis=0)
    labels=np.concatenate(labels,axis=0)

    return imgs,labels,ids

if __name__=="__main__":

    root='mnist'
    transform=transforms.ToTensor()
    train_dataset=torchvision.datasets.MNIST(root,train=True,download=True,transform=transform)
    val_dataset=torchvision.datasets.MNIST(root,train=False,download=True,transform=transform)

    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

    classes=[10]

    imgs_train,labels_train,ids_train = data_loader_to_array(train_data_loader)
    dump(os.path.join(DATA_DIR,TRAIN_DATASET_PKL),imgs_train,labels_train,ids_train, classes)

    imgs_val,labels_val,ids_val = data_loader_to_array(val_data_loader)
    dump(os.path.join(DATA_DIR,VAL_DATASET_PKL), imgs_val,labels_val,ids_val, classes)

    os.remove(os.path.join(DATA_DIR,TEST_DATASET_PKL))


