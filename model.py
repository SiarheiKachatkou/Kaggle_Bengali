import tensorflow as tf

from consts import IMG_W,IMG_H,N_CHANNELS

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

    def save(self,path_to_file):
        for m in range(len(self._models)):
            self._models[m].save_weights(path_to_file+'_{}'.format(m),save_format='tf')



