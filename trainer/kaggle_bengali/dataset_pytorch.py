import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from .consts import IMG_H,IMG_W
import glob
import os
import pickle

def _normalize_img(img):
    if len(img.shape)==2:
        img=np.expand_dims(img,axis=-1)
        img=np.concatenate([img,img,img],axis=2)
    else:
        if img.shape[-1]==1:
            img=np.concatenate([img,img,img],axis=2)

    eps=1e-3
    img=img.astype(np.float32)
    return (255-img)/255

class BengaliDataset(Dataset):
    def __init__(self, images, labels=None, transform_fn=None):
        self._images=[cv2.resize(im,(IMG_W,IMG_H)) for im in images]

        self._labels=labels
        self._transform_fn=transform_fn

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img=self._images[idx]
        label= -1 if self._labels is None else self._labels[idx]

        #img=np.concatenate([img,img,img],axis=-1)

        if self._transform_fn:
            img=self._transform_fn(img)

        img=_normalize_img(img)

        img_channel_first=np.transpose(img,[2,0,1])

        return {'image':img_channel_first,'label':label}



class BengaliDatasetOpt(Dataset):
    def __init__(self, local_dir, transform_fn=None):
        self._data_files=glob.glob(os.path.join(local_dir,'*'))

        self._transform_fn=transform_fn

        self._load_single_file(idx=0)

        self._length=len(self._images)*len(self._data_files)

        self._loaded_images_from_current_file=0

    def _load_single_file(self,idx=None):

        if idx is None:
            file_path=np.random.choice(self._data_files)
        else:
            file_path=self._data_files[idx]

        with open(file_path,'rb') as file:
            images,labels,ids, classes = pickle.load(file)

        self._images=[cv2.resize(im,(IMG_W,IMG_H)) for im in images]

        self._labels=labels

    def __len__(self):
        return self._length

    def __getitem__(self, idx):

        if self._loaded_images_from_current_file>=len(self._images):
            self._load_single_file()
            self._loaded_images_from_current_file=0
        else:
            self._loaded_images_from_current_file+=1

        idx=idx%len(self._images)

        img=self._images[idx]
        label= -1 if self._labels is None else self._labels[idx]

        #img=np.concatenate([img,img,img],axis=-1)

        if self._transform_fn:
            img=self._transform_fn(img)

        img=_normalize_img(img)

        img_channel_first=np.transpose(img,[2,0,1])

        return {'image':img_channel_first,'label':label}




