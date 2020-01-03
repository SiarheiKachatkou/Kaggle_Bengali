import pandas as pd
import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from consts import DATA_DIR,TRAIN_IMAGE_DATA_PATTERN, IMG_HEIGHT,IMG_WIDTH,N_CHANNELS,TRAIN_CSV,CLASS_MAP_CSV, IMG_H,IMG_W,BATCH_SIZE,EPOCHS,LR, DATASET_PKL,MODELS_DIR

for MODEL_TARGET in ['grapheme_root','vowel_diacritic','consonant_diacritic']:

    class_map=pd.read_csv(os.path.join(DATA_DIR,CLASS_MAP_CSV))
    CLASSES=sum(class_map['component_type']==MODEL_TARGET)

    if do_resave_dataset:
        train_csv=pd.read_csv(os.path.join(DATA_DIR,TRAIN_CSV))
        train_csv.set_index('image_id', inplace=True)

        img_data_files=glob.glob(os.path.join(DATA_DIR,TRAIN_IMAGE_DATA_PATTERN))

        image_ids=[]
        imgs=[]
        labels=[]

        #img_data_files=img_data_files[:1] #TODO test
        for filename in img_data_files:
            df_imgs = pd.read_parquet(filename,engine='pyarrow')
            df_imgs.set_index('image_id', inplace=True)
            image_ids.extend(list(df_imgs.index))
            def preproc(x):
                x=cv2.resize(np.reshape(x,[IMG_HEIGHT,IMG_WIDTH,N_CHANNELS]),(IMG_W,IMG_H))
                return np.expand_dims(x,axis=-1)
            images=[preproc(x) for x in list(df_imgs.values)]
            imgs.extend(images)
            labels.extend(list(np.array(train_csv.loc[df_imgs.index,MODEL_TARGET])))

        with open(os.path.join(DATA_DIR,DATASET_PKL+MODEL_TARGET),'wb') as file:
            pickle.dump([imgs,labels],file)
    else:
        with open(os.path.join(DATA_DIR,DATASET_PKL+MODEL_TARGET),'rb') as file:
            [imgs,labels]=pickle.load(file)

    print('{} train images loaded'.format(len(imgs)))



    ''''
    model=tf.keras.applications.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=(IMG_H,IMG_W,N_CHANNELS),
        pooling=None,
        classes=CLASSES)
    '''
    model=tf.keras.applications.MobileNet(
        input_shape=(IMG_H,IMG_W,N_CHANNELS),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=CLASSES)

    imgs=np.array(imgs)
    labels=np.array(labels)

    image_data_generator.fit(imgs)

    model.compile(optimizer=keras.optimizers.Adam(lr=LR),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(image_data_generator.flow(imgs,labels,batch_size=BATCH_SIZE),steps_per_epoch=len(imgs)/(BATCH_SIZE),epochs=EPOCHS)

    model.save_weights(os.path.join(DATA_DIR,MODELS_DIR,MODEL_TARGET),save_format='tf')

