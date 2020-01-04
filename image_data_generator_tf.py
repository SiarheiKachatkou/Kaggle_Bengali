import pickle
import tensorflow as tf
import tensorflow.keras as keras

class ImageDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def __init__(self, samplewise_center=True, samplewise_std_normalization=True):
        super(ImageDataGenerator, self).__init__(samplewise_center=samplewise_center,samplewise_std_normalization=samplewise_std_normalization)


    def save(self,filename):
        with open(filename,'wb') as file:
            pickle.dump(self,file)

    @classmethod
    def load(cls,filename):
        with open(filename,'rb') as file:
            return pickle.load(file)

#TODO add augmentation
