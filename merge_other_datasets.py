import numpy as np
import os
import cv2
from kagglebengali.consts import IMG_H,IMG_W,DATA_DIR,TRAIN_DATASET_DIR
from kagglebengali.create_dataset_utils import dump

def load_kuzushi(begin_label):
    p_imgs='/home/sergey/Downloads/handwriten_data/kuzushiji/k49-train-imgs/arr_0.npy'
    imgs=np.load(p_imgs,allow_pickle=True)

    p_labels='/home/sergey/Downloads/handwriten_data/kuzushiji/k49-train-labels/arr_0.npy'
    labs=np.load(p_labels,allow_pickle=True)

    images=[]
    labels=[]
    ids=[]


    for i,(img,label) in enumerate(zip(imgs,labs)):
        img=cv2.resize(img,(IMG_W,IMG_H))
        img=255-img
        images.append(img)
        labels.append([begin_label+label,0,0])
        ids.append(p_imgs+'_{}'.format(i))

    labels_added=np.max(labs)+1

    return images,labels,ids,labels_added




def load_malayan(begin_label):
    p='/home/sergey/Downloads/handwriten_data/malayalam-handwritten-characters/datasetgray.npy'
    data=np.load(p,allow_pickle=True)

    images=[]
    labels=[]
    ids=[]


    for i,(img,label) in enumerate(data):
        img=cv2.resize(img,(IMG_W,IMG_H))
        images.append(img)
        labels.append([begin_label+np.where(label==1)[0][0],0,0])
        labels_added=len(label)
        ids.append(p+'_{}'.format(i))

    return images,labels,ids,labels_added

begin_label=168
classes=[]


images,labels,ids,labels_added=load_kuzushi(begin_label)
dump(os.path.join(DATA_DIR,TRAIN_DATASET_DIR),images,labels,ids, classes,prefix='kuzushi',mode='append')

begin_label=begin_label+labels_added

images,labels,ids,labels_added=load_malayan(begin_label)
dump(os.path.join(DATA_DIR,TRAIN_DATASET_DIR),images,labels,ids, classes,prefix='malayan',mode='append')

print('total_labels={}'.format(begin_label+labels_added))
