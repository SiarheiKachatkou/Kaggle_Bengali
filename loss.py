import torch
import torch.nn as nn
import numpy as np


def calc_classes_weights(labels,classes_list):

    weights=[]
    classes_count=np.shape(labels)[1]
    for c in range(classes_count):
        l=labels[:,c]
        unique_labels=np.unique(l)
        assert len(unique_labels)==classes_list[c], "not all classes are presented in dataset"
        l=list(l)
        freq=[l.count(unique_label) for unique_label in unique_labels]
        w=[1./f for f in freq]
        weights.append(w)

    return weights

class RecallScore(nn.Module):
    def __init__(self,classes_weights):
        super().__init__()

        self.class_weights=classes_weights
        self.classes_labels=list(range(len(classes_weights)))
        self._raw_loss_fn=nn.CrossEntropyLoss()

    def forward(self,y_pred,y_true,):
        idxs=torch.where(y_true==self.classes_labels)
        w=self.class_weights[idxs]
        loss=self._raw_loss_fn(y_pred=y_pred,y_true=y_true,weights=w)
        return loss


