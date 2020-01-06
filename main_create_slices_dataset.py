import os
import numpy as np
import collections
from functools import partial
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataset_pytorch import BengaliDataset
from model import Model
from consts import DATA_DIR,MODEL_NAME, BATCH_SIZE, IMG_H, IMG_W, TEST_CSV, RAW_DIR, TRAIN_DATASET_PKL, VAL_DATASET_PKL, TEST_DATASET_PKL, IMAGE_GEN_PKL, MODELS_DIR, SUBMISSION_DIR, SUBMISSION_CSV,SAMPLE_SUBMISSION_CSV

from create_dataset_utils import load

import sys
sys.path.append('../DNNDebug/Python')
from dnn_slice import DNNSlice
sys.path.append('..')
from DNNDebug.Python.symbol import Symbol

if __name__ == "__main__":

    activation_name='_block_21._r'

    imgs, labels, ids, classes = load(os.path.join(DATA_DIR,VAL_DATASET_PKL))

    model=Model()
    model.load(os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME),classes)

    dataset=BengaliDataset(imgs,labels=None)

    dataloader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    for name, m in model.named_modules():
        if type(m)==nn.ReLU:
            # partial to assign the layer name to each hook
            m.register_forward_hook(partial(save_activation, name))

    predicted_acts_batches=[]
    preds_batches=[]
    for batch in dataloader:
        imgs_normalized=batch['image']
        preds_batches.append(model._predict_on_tensor(imgs_normalized))
        act=activations[activation_name]
        predicted_acts_batches.append(act.data.numpy())

    preds=np.concatenate(preds_batches,axis=0)
    acts=np.concatenate(predicted_acts_batches,axis=0)

    slices=[]
    for img,image_id,predicted_label, act, true_label in zip(imgs,ids,preds,acts, labels):

        slice=DNNSlice(activation_slices_list=[act], activation_names=[activation_name],
                       img=img, img_slice=img, img_slice_begin=0,
                       img_slice_end=IMG_W,
                       symbol=Symbol(label=predicted_label, x=0, prob=1.0,#TODO take actual probs
                                     true_label=true_label), full_true_label=true_label, full_predicted_label=predicted_label,
                       image_id=image_id,
                       grads=None)
        slices.append(slice)
