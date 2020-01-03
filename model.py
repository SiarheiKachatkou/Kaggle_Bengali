import tensorflow as tf
import numpy as np
from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE

class Model():
    def __init__(self):
        self._models=[]


    def _get_single_model(self, classes):

        ''''
        model=tf.keras.applications.ResNet50(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=(IMG_H,IMG_W,N_CHANNELS),
            pooling=None,
            classes=CLASSES)
        '''

        model=tf.keras.applications.MobileNet(
        input_shape=(IMG_H,IMG_W,N_CHANNELS),
        alpha=1.0,
        depth_multiplier=1,
        dropout=0.001,
        include_top=True,
        weights=None,
        input_tensor=None,
        pooling=None,
        classes=classes)

        return model

    def compile(self,classes_list,**kwargs):

        for classes in classes_list:
            model=self._get_single_model(classes)
            model.compile(**kwargs)
            self._models.append(model)


    def fit_generator(self,gen,train_images,train_labels, val_images, val_labels, batch_size,epochs):
        for m in range(len(self._models)):
            self._models[m].fit_generator(gen.flow(train_images,train_labels[:,m]),validation_data=gen.flow(val_images,val_labels[:,m]),steps_per_epoch=len(train_images)/(batch_size),epochs=epochs)

    def _get_model_filename(self,path_to_file,index):
        return path_to_file+'_{}'.format(index)

    def save(self,path_to_file):
        for m in range(len(self._models)):
            self._models[m].save_weights(self._get_model_filename(path_to_file, m),save_format='tf')

    def load(self,path_fo_file,classes_list):

        self._models=[]
        for index in range(len(classes_list)):
            model_file_path=self._get_model_filename(path_fo_file,index)
            model=self._get_single_model(classes_list[index])
            model.load_weights(model_file_path)
            self._models.append(model)

    def predict(self, gen, images):

        pred_labels=[]
        flow=gen.flow(images,shuffle=False, batch_size=BATCH_SIZE)
        max_batch_idx=len(flow)
        for batch_idx, img_batch in enumerate(flow):
            predictions_batch=[m.predict(img_batch) for m in self._models]
            labels_batch=[np.argmax(p,axis=-1) for p in predictions_batch]
            labels=np.stack(labels_batch,axis=-1)
            pred_labels.append(labels)

            if batch_idx==max_batch_idx-1:
                break

        pred_labels=np.concatenate(pred_labels,axis=0)
        return pred_labels






