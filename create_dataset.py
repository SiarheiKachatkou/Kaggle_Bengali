import pandas as pd
import os
import glob
import cv2
import numpy as np
import sklearn.model_selection
from create_dataset_utils import dump,load

from consts import DATA_DIR,RAW_DIR,TRAIN_IMAGE_DATA_PATTERN, TEST_IMAGE_DATA_PATTERN, \
    IMG_HEIGHT,IMG_WIDTH,N_CHANNELS,TRAIN_CSV,CLASS_MAP_CSV, IMG_H,IMG_W, \
    TRAIN_DATASET_DIR, VAL_DATASET_DIR, TEST_DATASET_DIR, \
    IMAGE_GEN_PKL, SEED, TARGETS

debug_mode=False

def load_parquet(path_pattern, labels_csv):

    image_ids=[]
    imgs=[]
    labels=[]
    img_data_files=glob.glob(path_pattern)
    if debug_mode:
        img_data_files=img_data_files[:1]

    for filename in img_data_files:
        df_imgs = pd.read_parquet(filename,engine='pyarrow')
        df_imgs.set_index('image_id', inplace=True)
        image_ids.extend(list(df_imgs.index))

        images=[np.reshape(x,[IMG_HEIGHT,IMG_WIDTH,N_CHANNELS]) for x in list(df_imgs.values)]
        imgs.extend(images)
        if labels_csv is not None:
            labels.extend(list(np.array(labels_csv.loc[df_imgs.index])[:,:-1]))
        else:
            labels=None

    imgs=np.array(imgs)
    if labels is not None:
        labels=np.array(labels).astype(np.int)
    image_ids=np.array(image_ids)
    return imgs,image_ids,labels


if __name__=="__main__":

    class_map=pd.read_csv(os.path.join(DATA_DIR,RAW_DIR, CLASS_MAP_CSV))
    train_csv=pd.read_csv(os.path.join(DATA_DIR,RAW_DIR, TRAIN_CSV))
    train_csv.set_index('image_id', inplace=True)

    classes = [sum(class_map['component_type']==target) for target in TARGETS]

    imgs,image_ids,labels = load_parquet(os.path.join(DATA_DIR,RAW_DIR, TRAIN_IMAGE_DATA_PATTERN),train_csv)

    skf = sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    #TODO place for improvements
    labels_0=labels[:,0]
    train_index, test_index = next(skf.split(imgs, labels_0))

    imgs_train, labels_train,ids_train = imgs[train_index], labels[train_index], image_ids[train_index]

    imgs_val, labels_val, ids_val =imgs[test_index], labels[test_index], image_ids[test_index]

    dump(os.path.join(DATA_DIR,TRAIN_DATASET_DIR),imgs_train,labels_train,ids_train, classes)

    dump(os.path.join(DATA_DIR,VAL_DATASET_DIR), imgs_val,labels_val,ids_val, classes)

    test_imgs,test_image_ids,test_labels = load_parquet(os.path.join(DATA_DIR, RAW_DIR, TEST_IMAGE_DATA_PATTERN),labels_csv=None)
    dump(os.path.join(DATA_DIR,TEST_DATASET_DIR), test_imgs,test_labels,test_image_ids, classes)


