import os
import numpy as np
import collections
import argparse
import pandas as pd
from functools import partial
import re
import torch
import shutil
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from dataset_pytorch import BengaliDataset
from model import Model
from consts import DATA_DIR,MODEL_NAME, TARGETS, BATCH_SIZE, IMG_H, IMG_W, TEST_CSV, RAW_DIR, TRAIN_DATASET_PKL, VAL_DATASET_PKL, TEST_DATASET_PKL, IMAGE_GEN_PKL, MODELS_DIR, SUBMISSION_DIR, SUBMISSION_CSV,SAMPLE_SUBMISSION_CSV

from create_dataset_utils import load

import sys
sys.path.append('../DNNDebug/Python')
from dnn_slice import DNNSlice, write_slices
from dnn_symbol import Symbol

def list_label_to_symbol(class_map,list_label):

    s=''
    min_length=min(len(TARGETS),len(list_label))
    for idx in range(min_length):
        c,l = TARGETS[idx], list_label[idx]
        df=class_map[class_map['component_type']==c]
        new_char=df.iloc[l,-1]
        if new_char!='0':
            s+=str(new_char)
    return s

if __name__ == "__main__":


    parser=argparse.ArgumentParser()
    parser.add_argument('--sub_dataset',type=str,default='test')
    parser.add_argument('--tag',type=str,default='174e3c')
    parser.add_argument('--dst_dir',type=str,default='/home/sergey/1T/DNNDebug/Data/SlicesDataset/')
    parser.add_argument('--max_imgs_count',type=int,default=100)
    parser.add_argument('--activation_name_postfix',type=str,default='conv3')
    parser.add_argument('--slices_per_file',type=int,default=100)
    parser.add_argument('--class_map_path',type=str)

    args=parser.parse_args()
    sub_dataset=args.sub_dataset
    dst_dir=args.dst_dir
    tag=args.tag
    max_imgs_count=args.max_imgs_count
    activation_name_postfix=args.activation_name_postfix
    slices_per_file=args.slices_per_file

    dataset_pkl=TRAIN_DATASET_PKL if sub_dataset=='train' else VAL_DATASET_PKL

    class_map=pd.read_csv(args.class_map_path)

    imgs, labels, ids, classes = load(os.path.join(DATA_DIR,dataset_pkl))
    if max_imgs_count<len(imgs):
        idxs=np.random.choice(range(len(imgs)),max_imgs_count,replace=False)
        imgs=imgs[idxs]
        labels=labels[idxs]
        ids=ids[idxs]
    print('{} images loaded'.format(len(imgs)))

    model=Model()
    model.load(os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME),classes)

    dataset=BengaliDataset(imgs,labels)

    dataloader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

    activations = collections.defaultdict(list)
    def save_activation(name, mod, inp, out):
        activations[name].append(out.cpu())

    activation_names=[]
    for name, m in model.named_modules():
        print('possible activations name {}'.format(name))
        # partial to assign the layer name to each hook
        if re.match('.*block_([0-9]*)$',name) is not None:
            activation_names.append(name)
            m.register_forward_hook(partial(save_activation, name))

    dataset_dir=os.path.join(dst_dir,sub_dataset+'_{}'.format(tag))
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.mkdir(dataset_dir)

    def dump_slices(slices,img_idx,batch_idx):
        if len(slices)>0:
            filename = os.path.join(dataset_dir,'Bengali_{}_{}.bin'.format(img_idx,batch_idx))
            write_slices(slices,filename)
            print('{} slices saved to {}'.format(len(slices),filename))



    slices=[]
    global_img_idx=0
    for bathc_idx, batch in tqdm(enumerate(dataloader)):

        imgs_normalized=batch['image']
        true_labels=batch['label'].cpu().numpy()

        preds_batch=model._predict_on_tensor(imgs_normalized)
        acts_list=[]
        for name_idx, name in enumerate(activation_names):
            act=activations[name]
            act_batch=act[0].data.cpu().numpy()
            acts_list.append(act_batch)

        activations=collections.defaultdict(list)

        img_idx=0
        imgs=imgs_normalized.cpu().numpy()[:,0,:,:]
        for img,predicted_label, true_label in zip(imgs,preds_batch,true_labels):

            pred_symbol=list_label_to_symbol(class_map,predicted_label)
            true_symbol=list_label_to_symbol(class_map,true_label)

            activation_slices_list=[act[img_idx].flatten() for act in acts_list]
            slice=DNNSlice(activation_slices_list=activation_slices_list, activation_names=activation_names,
                           img=None, img_slice=img, img_slice_begin=0,
                           img_slice_end=IMG_W,
                           symbol=Symbol(label=pred_symbol, x=0, prob=1.0,#TODO take actual probs
                                 true_label=true_symbol), full_true_label=None, full_predicted_label=None,
                           image_id=ids[global_img_idx],
                           grads=None)
            slices.append(slice)
            img_idx+=1
            global_img_idx+=1

            if len(slices)>slices_per_file:
                dump_slices(slices,img_idx,bathc_idx)
                slices=[]

        if len(slices)>0:
            dump_slices(slices,img_idx,bathc_idx)
            slices=[]
