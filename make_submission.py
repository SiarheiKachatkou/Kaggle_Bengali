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

if __name__ == "__main__":

    with open(os.path.join(DATA_DIR,TEST_DATASET_PKL),'rb') as file:
        train_images, train_labels, classes = pickle.load(file)
