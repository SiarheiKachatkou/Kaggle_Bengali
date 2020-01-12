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

def augment_top_is_matter(label_a,label_b,imgs_train,labels_train,aug_count):

    imgs_train_ab,labels_train_ab,ids_train_ab=[],[],[]
    imgs_a=imgs_train[(labels_train==label_a).flatten()]
    imgs_b=imgs_train[(labels_train==label_b).flatten()]
    for a_idx in range(aug_count):
        img_a=random.choice(imgs_a)
        img_b=random.choice(imgs_b)
        h=len(img_a)
        half_h=h//2
        img_aug_a=np.concatenate([img_a[:half_h],img_b[half_h:]],axis=0)
        img_aug_b=np.concatenate([img_b[:half_h],img_a[half_h:]],axis=0)
        imgs_train_ab.append(img_aug_a)
        labels_train_ab.append([label_a])
        ids_train_ab.append(a_idx)
        imgs_train_ab.append(img_aug_b)
        labels_train_ab.append([label_b])
        ids_train_ab.append(a_idx)
    return np.array(imgs_train_ab), np.array(labels_train_ab),np.array(ids_train_ab)

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
    imgs_augm=[]
    labels_augm=[]
    idxs_augm=[]
    for label_a,label_b in [[5,3],[9,4],[7,9]]:
        imgs_train_ab,labels_train_ab,ids_train_ab=augment_top_is_matter(label_a,label_b,imgs_train,labels_train,aug_count=100)
        imgs_augm.append(imgs_train_ab)
        labels_augm.append(labels_train_ab)
        idxs_augm.append(ids_train_ab)

    imgs_train=np.concatenate([imgs_train]+imgs_augm,axis=0)
    labels_train=np.concatenate([labels_train]+labels_augm,axis=0)
    ids_train=np.concatenate([ids_train]+idxs_augm,axis=0)

    z=list(zip(imgs_train,labels_train,ids_train))
    random.shuffle(z)
    imgs_train,labels_train,ids_train=zip(*z)

    dump(os.path.join(DATA_DIR,TRAIN_DATASET_PKL),imgs_train,labels_train,ids_train, classes)

    imgs_val,labels_val,ids_val = data_loader_to_array(val_data_loader)
    dump(os.path.join(DATA_DIR,VAL_DATASET_PKL), imgs_val,labels_val,ids_val, classes)

    dump(os.path.join(DATA_DIR,TEST_DATASET_PKL), imgs_val,labels_val,ids_val, classes)


