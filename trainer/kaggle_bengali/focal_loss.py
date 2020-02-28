import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, inputs, targets):

        cross_entropy = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-cross_entropy)
        F_loss = self.alpha * (1-pt)**self.gamma * cross_entropy

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
