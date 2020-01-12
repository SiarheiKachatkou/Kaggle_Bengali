import pandas as pd
import os
import glob
import cv2
import random
import torch
import torchvision
from torchvision import transforms
import numpy as np
import sklearn.model_selection
from create_dataset_utils import dump,load

from consts import BATCH_SIZE, DATA_DIR,RAW_DIR,TRAIN_IMAGE_DATA_PATTERN, TEST_IMAGE_DATA_PATTERN, IMG_HEIGHT,IMG_WIDTH,N_CHANNELS,TRAIN_CSV,CLASS_MAP_CSV, IMG_H,IMG_W, TRAIN_DATASET_PKL, VAL_DATASET_PKL, TEST_DATASET_PKL, IMAGE_GEN_PKL, SEED, TARGETS

debug_mode=False

def preproc(x):
    x=np.squeeze(x,axis=0)
    x=cv2.resize(x,(IMG_W,IMG_H))
    return np.expand_dims(x,axis=-1)

def data_loader_to_array(data_loader):
    imgs=[]
    labels=[]
    ids=[]
    img_idx=0
    for batch_img,batch_label in data_loader:

        imgs.extend([preproc(img) for img in batch_img.cpu().numpy()])
        labels.extend([[l] for l in batch_label.cpu().numpy()])
        for _ in range(len(batch_img)):
            ids.append(img_idx)
            img_idx+=1

    return np.array(imgs),np.array(labels),np.array(ids)

def augment53(imgs_train,labels_train,aug_count):

    imgs_train_53,labels_train_53,ids_train_53=[],[],[]
    imgs_5=imgs_train[(labels_train==5).flatten()]
    imgs_3=imgs_train[(labels_train==3).flatten()]
    for a_idx in range(aug_count):
        img_5=random.choice(imgs_5)
        img_3=random.choice(imgs_3)
        h=len(img_5)
        half_h=h//2
        img_aug_3=np.concatenate([img_3[:half_h],img_5[half_h:]],axis=0)
        img_aug_5=np.concatenate([img_5[:half_h],img_3[half_h:]],axis=0)
        imgs_train_53.append(img_aug_3)
        labels_train_53.append([3])
        ids_train_53.append(a_idx)
        imgs_train_53.append(img_aug_5)
        labels_train_53.append([5])
        ids_train_53.append(a_idx)
    return np.array(imgs_train_53), np.array(labels_train_53),np.array(ids_train_53)



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
    imgs_train_53,labels_train_53,ids_train_53=augment53(imgs_train,labels_train,aug_count=1000)

    imgs_train=np.concatenate([imgs_train,imgs_train_53],axis=0)
    labels_train=np.concatenate([labels_train,labels_train_53],axis=0)
    ids_train=np.concatenate([ids_train,ids_train_53],axis=0)

    dump(os.path.join(DATA_DIR,TRAIN_DATASET_PKL),imgs_train,labels_train,ids_train, classes)

    imgs_val,labels_val,ids_val = data_loader_to_array(val_data_loader)
    dump(os.path.join(DATA_DIR,VAL_DATASET_PKL), imgs_val,labels_val,ids_val, classes)

    dump(os.path.join(DATA_DIR,TEST_DATASET_PKL), imgs_val,labels_val,ids_val, classes)


