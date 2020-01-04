import torch
import torchvision
import numpy as np
from model_base import ModelBase
from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR


class Model(ModelBase, torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self.backbone=torchvision.models.resnet18(pretrained=False)


    def compile(self,classes_list,**kwargs):
        raise NotImplementedError


    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, **kwargs):
        raise NotImplementedError

    def save(self,path_to_file):
        raise NotImplementedError

    def load(self,path_fo_file,classes_list):
        raise NotImplementedError

    def predict(self, images):
        raise NotImplementedError

