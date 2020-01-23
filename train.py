import os
import numpy as np
import argparse
import pandas as pd
from model import Model
from create_dataset_utils import load
from score import calc_score
from consts import MODELS_PRETRAINED_DIR, DATA_DIR,MODEL_NAME,BATCH_SIZE,EPOCHS, TRAIN_DATASET_PKL, VAL_DATASET_PKL, MODELS_DIR, METRIC_FILE_PATH

debug_regime=False

if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--fine_tune',type=int,default=0)
    args=parser.parse_args()

    train_images, train_labels, _, classes = load(os.path.join(DATA_DIR,TRAIN_DATASET_DIR))
    val_images, val_labels, _, _  = load(os.path.join(DATA_DIR,VAL_DATASET_DIR))

    if debug_regime:
        max_samples=100
        train_images=train_images[:max_samples]
        train_labels=train_labels[:max_samples]
        val_images=val_images[:max_samples]
        val_labels=val_labels[:max_samples]

    print('{} train images loaded'.format(len(train_images)))
    print('{} val images loaded'.format(len(val_images)))

    model_dir=os.path.join(DATA_DIR,MODELS_DIR)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_filepath=os.path.join(model_dir,MODEL_NAME)


    model=Model()

    model.compile(classes_list=classes)
    if args.fine_tune!=0:
        model_pretrained_filepath=os.path.join(DATA_DIR,MODELS_PRETRAINED_DIR,MODEL_NAME)
        model.load(model_pretrained_filepath, classes)
    model.fit(train_images,train_labels, val_images,val_labels,path_to_file=model_filepath,batch_size=BATCH_SIZE,epochs=EPOCHS)

    model.save(model_filepath)

    model_loaded=Model()
    model_loaded.load(model_filepath, classes)
    model_loaded.eval()
    val_preds=model_loaded.predict( val_images)
    acc=[np.equal(val_pred,val_label) for val_pred,val_label in zip(val_preds,val_labels)]
    acc=np.array(acc,dtype=np.float32)
    acc=np.mean(acc,axis=0)
    print('validation accuracy of loaded model = {}'.format(acc))

    score=calc_score(solution=val_preds,submission=val_labels)
    print('score={}'.format(score))
    with open(METRIC_FILE_PATH,'wt') as file:
        file.write(str(score))







