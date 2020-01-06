import os

from model import Model
from consts import DATA_DIR,MODEL_NAME, TEST_CSV, RAW_DIR, TRAIN_DATASET_PKL, TEST_DATASET_PKL, IMAGE_GEN_PKL, MODELS_DIR, SUBMISSION_DIR, SUBMISSION_CSV,SAMPLE_SUBMISSION_CSV
from create_dataset_utils import load

import sys
sys.path.append('../DNNDebug/Python')
from dnn_slice import DNNSlice

if __name__ == "__main__":

    imgs, labels, ids, classes = load(os.path.join(DATA_DIR,TRAIN_DATASET_PKL))

    model=Model()
    model.load(os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME),classes)
    preds = model.predict(imgs)

    slices=[]
    for img,image_id,pred in zip(imgs,ids,preds):

        slice=DNNSlice(activation_slices_list=activation_slices, activation_names=activation_names,
                       img=img, img_slice=img_slice, img_slice_begin=frame_x,
                       img_slice_end=frame_x+FRAME_WIDTH,
                       symbol=symbol, full_true_label=true_label, full_predicted_label=predicted_label,
                       image_id=image_id,
                       grads=None)
        slices.append(slice)
