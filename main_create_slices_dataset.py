import os
import numpy as np
import collections
import argparse
import pandas as pd
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
from dnn_slice import DNNSlice, write_slices
from dnn_symbol import Symbol

def list_label_to_symbol(class_map,list_label):

    s=''
    for c,l in zip(['grapheme_root','vowel_diacritic','consonant_diacritic'],list_label):
        df=class_map[class_map['component_type']==c]
        new_char=df.iloc[l,-1]
        if new_char!='0':
            s+=new_char
    return s

if __name__ == "__main__":

    activation_name='_backbone.features.18.0'
    max_imgs_count=1000

    parser=argparse.ArgumentParser()
    parser.add_argument('--sub_dataset',type=str,default='test')
    parser.add_argument('--tag',type=str,default='c080f40')
    parser.add_argument('--dst_dir',type=str,default='/home/sergey/1T/DNNDebug/Data/SlicesDataset/')
    args=parser.parse_args()
    sub_dataset=args.sub_dataset
    dst_dir=args.dst_dir
    tag=args.tag

    dataset_pkl=TRAIN_DATASET_PKL if sub_dataset=='train' else VAL_DATASET_PKL

    class_map=pd.read_csv('data/raw/class_map.csv')

    imgs, labels, ids, classes = load(os.path.join(DATA_DIR,dataset_pkl))
    if max_imgs_count<len(imgs):
        imgs=imgs[:max_imgs_count]
        labels=labels[:max_imgs_count]
        ids=ids[:max_imgs_count]
    print('{} images loaded'.format(len(imgs)))

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
        if type(m)==nn.Conv2d:
            # partial to assign the layer name to each hook
            print('possible activations name {}'.format(name))
            m.register_forward_hook(partial(save_activation, name))


    predicted_acts_batches=[]
    preds_batches=[]
    for batch in dataloader:
        imgs_normalized=batch['image']
        preds_batches.append(model._predict_on_tensor(imgs_normalized))
        act=activations[activation_name]
        predicted_acts_batches.append(act[0].data.cpu().numpy())
        activations=collections.defaultdict(list)

    preds=np.concatenate(preds_batches,axis=0)
    acts=np.concatenate(predicted_acts_batches,axis=0)

    slices=[]
    for img,image_id,predicted_label, act, true_label in zip(imgs,ids,preds,acts, labels):

        pred_symbol=list_label_to_symbol(class_map,predicted_label)
        true_symbol=list_label_to_symbol(class_map,true_label)

        slice=DNNSlice(activation_slices_list=[act], activation_names=[activation_name],
                       img=None, img_slice=img, img_slice_begin=0,
                       img_slice_end=IMG_W,
                       symbol=Symbol(label=pred_symbol, x=0, prob=1.0,#TODO take actual probs
                             true_label=true_symbol), full_true_label=None, full_predicted_label=None,
                       image_id=image_id,
                       grads=None)
        slices.append(slice)

    dir=os.path.join(dst_dir,sub_dataset+'_{}'.format(tag))
    if not os.path.exists(dir):
        os.mkdir(dir)
    filename=os.path.join(dir,'Bengali.bin')
    write_slices(slices,filename)
    print('{} slices saved to {}'.format(len(slices),filename))
