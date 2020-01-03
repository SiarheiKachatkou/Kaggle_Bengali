import pandas as pd
import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import pickle
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
from consts import DATA_DIR,TRAIN_IMAGE_DATA_PATTERN, IMG_HEIGHT,IMG_WIDTH,N_CHANNELS,TRAIN_CSV,CLASS_MAP_CSV, IMG_H,IMG_W,BATCH_SIZE,EPOCHS,LR, TRAIN_DATASET_PKL, VAL_DATASET_PKL,SEED


def preproc(x):
    x=cv2.resize(np.reshape(x,[IMG_HEIGHT,IMG_WIDTH,N_CHANNELS]),(IMG_W,IMG_H))
    return np.expand_dims(x,axis=-1)


if __name__=="__main__":

    class_map=pd.read_csv(os.path.join(DATA_DIR,CLASS_MAP_CSV))
    train_csv=pd.read_csv(os.path.join(DATA_DIR,TRAIN_CSV))
    train_csv.set_index('image_id', inplace=True)
    img_data_files=glob.glob(os.path.join(DATA_DIR,TRAIN_IMAGE_DATA_PATTERN))


    TARGETS=['grapheme_root','vowel_diacritic','consonant_diacritic']
    classes = [sum(class_map['component_type']==target) for target in TARGETS]

    image_ids=[]
    imgs=[]
    labels=[]
    #img_data_files=img_data_files[:1] #TODO test
    for filename in img_data_files:
        df_imgs = pd.read_parquet(filename,engine='pyarrow')
        df_imgs.set_index('image_id', inplace=True)
        image_ids.extend(list(df_imgs.index))

        images=[preproc(x) for x in list(df_imgs.values)]
        imgs.extend(images)
        labels.extend(list(np.array(train_csv.loc[df_imgs.index])[:,:-1]))

    imgs=np.array(imgs)
    labels=np.array(labels).astype(np.int)

    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    #TODO place for improvements
    labels_0=labels[:,0]
    train_index, test_index = next(skf.split(imgs, labels_0))

    imgs_train, labels_train =imgs[train_index], labels[train_index]
    imgs_val, labels_val =imgs[test_index], labels[test_index]

    with open(os.path.join(DATA_DIR,TRAIN_DATASET_PKL),'wb') as file:
        pickle.dump([imgs_train,labels_train,classes],file)

    with open(os.path.join(DATA_DIR,VAL_DATASET_PKL),'wb') as file:
        pickle.dump([imgs_val,labels_val,classes],file)

