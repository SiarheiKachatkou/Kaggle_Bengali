import os
import pickle
import shutil
import cv2
import numpy as np
from .consts import IMG_H, IMG_W,TOP_CUT,LEFT_CUT,PAD

def crop_symbol(img):

    img=img[TOP_CUT:-TOP_CUT,LEFT_CUT:-LEFT_CUT]

    _,bin=cv2.threshold(img,0,255,cv2.THRESH_OTSU)
    black=np.where(bin<100)
    left,right = min(black[1]), max(black[1])
    top,bottom = min(black[0]), max(black[0])
    img=img[top:bottom,left:right]

    # padding for do not change aspect ratio
    height=bottom-top
    width=right-left
    max_size=max(height,width)+PAD

    img=np.pad(img,[((max_size-height)//2,),((max_size-width)//2,),(0,)],mode='constant',constant_values=255)

    img=cv2.resize(img,(IMG_W,IMG_H))

    return img

def dump(path_to_dir, imgs,labels, ids, classes, max_img_per_file=10000):

    if os.path.exists(path_to_dir):
        shutil.rmtree(path_to_dir)
    os.mkdir(path_to_dir)

    idxs=range(0,len(imgs),max_img_per_file)
    for i,idx in enumerate(idxs):
        end_idx=min(idx+max_img_per_file,len(imgs))

        the_slice=slice(idx,end_idx)
        path_to_file=os.path.join(path_to_dir,'{}.pkl'.format(i))
        with open(path_to_file,'wb') as file:
            imgs_preproc=np.array([crop_symbol(im) for im in imgs[the_slice]])
            the_labels=None if labels is None else labels[the_slice]
            pickle.dump([imgs_preproc,the_labels,ids[the_slice], classes],file)

def load(path_to_dir):

    _,_,files=next(os.walk(path_to_dir))

    imgs_all=[]
    labels_all=[]
    ids_all=[]
    for path_to_file in files:
        with open(os.path.join(path_to_dir,path_to_file),'rb') as file:
            imgs,labels,ids, classes = pickle.load(file)
            imgs_all.append(imgs)
            labels_all.append(labels)
            ids_all.append(ids)

    imgs_all=np.concatenate(imgs_all,axis=0)
    labels_all=np.concatenate(labels_all,axis=0)
    ids_all=np.concatenate(ids_all,axis=0)

    return imgs_all,labels_all,ids_all, classes
