import torch
import torch.nn as nn
import numpy as np
from .consts import BETA
from .focal_loss import FocalLoss

def calc_classes_weights(labels,classes_list):

    weights=[]
    classes_count=np.shape(labels)[1]
    for c in range(classes_count):
        l=labels[:,c]
        unique_labels=np.unique(l)
        assert len(unique_labels)==classes_list[c], "not all classes are presented in dataset"
        l=list(l)
        freq=[l.count(unique_label) for unique_label in unique_labels]

        #w=[(1-BETA)/(1-BETA**f) for f in freq]
        w=[1/f for f in freq]
        #total=sum(w)
        #w=[w_i/total for w_i in w]
        weights.append(w)

    return weights

class RecallScore(nn.Module):
    def __init__(self,classes_weights):
        super().__init__()
        self.class_weights=torch.Tensor(classes_weights).to(device=torch.device('cuda'))
        self.classes_labels=list(range(len(classes_weights)))
        self._raw_loss_fn=FocalLoss()#nn.CrossEntropyLoss()#weight=self.class_weights)

    def forward(self,output,target,):
        loss=self._raw_loss_fn(output,target)
        return loss


