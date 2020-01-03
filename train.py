import pandas as pd
import os
import glob
import tensorflow as tf
import tensorflow.keras as keras
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from model import Model
from image_data_generator import ImageDataGenerator
from consts import DATA_DIR,MODEL_NAME,BATCH_SIZE,EPOCHS,LR, TRAIN_DATASET_PKL, VAL_DATASET_PKL, MODELS_DIR

debug_regime=False

if __name__ == "__main__":

    with open(os.path.join(DATA_DIR,TRAIN_DATASET_PKL),'rb') as file:
        train_images, train_labels, classes = pickle.load(file)

    with open(os.path.join(DATA_DIR,VAL_DATASET_PKL),'rb') as file:
        val_images, val_labels, _ = pickle.load(file)

    if debug_regime:
        max_samples=100
        train_images=train_images[:max_samples]
        train_labels=train_labels[:max_samples]
        val_images=val_images[:max_samples]
        val_labels=val_labels[:max_samples]

    print('{} train images loaded'.format(len(train_images)))
    print('{} val images loaded'.format(len(val_images)))

    gen=ImageDataGenerator()
    gen.fit(train_images)

    model=Model()

    model.compile(classes_list=classes, optimizer=keras.optimizers.Adam(lr=LR),loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit_generator(gen,train_images,train_labels, val_images,val_labels,batch_size=BATCH_SIZE,epochs=EPOCHS)

    model_filepath=os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME)
    model.save(model_filepath)

    model_loaded=Model()
    model_loaded.load(model_filepath, classes)

    val_preds=model_loaded.predict(gen, val_images)
    acc=[float(all(val_pred==val_label)) for val_pred,val_label in zip(val_preds,val_labels)]
    print('validation accuracy of loaded model = {}'.format(np.mean(acc)))






