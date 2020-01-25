
class ModelBase():
    def __init__(self):
        pass

    def compile(self,classes_list,**kwargs):
        raise NotImplementedError


    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, path_to_model_save, **kwargs):
        raise NotImplementedError

    def save(self,path_to_file):
        raise NotImplementedError

    def load(self,path_fo_file,classes_list):
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError
