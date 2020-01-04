import os
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from model import Model
from create_dataset import load
from image_data_generator import ImageDataGenerator
from consts import DATA_DIR,MODEL_NAME,BATCH_SIZE,EPOCHS,LR, TRAIN_DATASET_PKL, VAL_DATASET_PKL, IMAGE_GEN_PKL, MODELS_DIR

debug_regime=False

if __name__ == "__main__":

    train_images, train_labels, _, classes = load(os.path.join(DATA_DIR,TRAIN_DATASET_PKL))
    val_images, val_labels, _, _  = load(os.path.join(DATA_DIR,VAL_DATASET_PKL))

    if debug_regime:
        max_samples=100
        train_images=train_images[:max_samples]
        train_labels=train_labels[:max_samples]
        val_images=val_images[:max_samples]
        val_labels=val_labels[:max_samples]

    print('{} train images loaded'.format(len(train_images)))
    print('{} val images loaded'.format(len(val_images)))

    gen=ImageDataGenerator.load(os.path.join(DATA_DIR,IMAGE_GEN_PKL))

    model=Model()

    model.compile(classes_list=classes)
    model.fit_generator(gen,train_images,train_labels, val_images,val_labels,batch_size=BATCH_SIZE,epochs=EPOCHS)

    model_filepath=os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME)
    model.save(model_filepath)

    model_loaded=Model()
    model_loaded.load(model_filepath, classes)

    val_preds=model_loaded.predict(gen, val_images)
    acc=[float(all(np.equal(val_pred,val_label))) for val_pred,val_label in zip(val_preds,val_labels)]
    acc=np.mean(acc)
    print('validation accuracy of loaded model = {}'.format(acc))
    with open('metric.txt','wt') as file:
        file.write(str(acc))







