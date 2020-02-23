import torch
import torchvision
import numpy as np
from tqdm import tqdm
from model_base import ModelBase
from score import calc_score
from dataset_pytorch import BengaliDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
import pretrainedmodels
from shake_shake_my import ShakeShake
import cv2
from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS, AUGM_PROB,FAST_PROTO_SCALE, \
    DROPOUT_P, LOSS_WEIGHTS, LR_SCHEDULER_PATINCE,USE_FREQ_SAMPLING
from loss import calc_classes_weights, RecallScore
from torch.utils.data import WeightedRandomSampler


class ConvBnRelu(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1,dilation=1):
        super().__init__()
        self._conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=kernel_size//2,dilation=dilation)
        self._bn=nn.BatchNorm2d(num_features=out_channels)
        self._r=nn.ReLU()
        self.out_features=out_channels

    def forward(self,x):
        x=self._conv(x)
        x=self._bn(x)
        x=self._r(x)
        return x

class ResNetBasicBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=k(3),stride=1,padding=1)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=k(3),stride=1,padding=1)
        self._bn=nn.BatchNorm2d(num_features=in_channels)
        self._r2=nn.ReLU()

    def forward(self,x):

        skip=x
        x=self._c1(x)
        x=self._r1(x)
        x=self._c2(x)
        x=self._bn(x)
        x=x+skip
        x=self._r2(x)

        return x

class ResNetBottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._in_channels=in_channels
        bottleneck_depth=in_channels//4
        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_depth,kernel_size=1,stride=1)
        self._bn1=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,kernel_size=k(3),stride=1,padding=k(3)//2)
        self._bn2=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r2=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=in_channels,kernel_size=1,stride=1)
        self._bn3=nn.BatchNorm2d(num_features=in_channels)
        self._r3=nn.ReLU()

    def forward(self,x):
        skip=x
        x=self._c1(x)
        x=self._r1(x)
        x=self._bn1(x)
        x=self._c2(x)
        x=self._r2(x)
        x=self._bn2(x)
        x=self._c3(x)
        x=self._bn3(x)

        x=x+skip
        x=self._r3(x)
        return x

class SEResNetBottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._in_channels=in_channels
        self.out_features=in_channels
        bottleneck_depth=in_channels//4
        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_depth,kernel_size=1,stride=1)
        self._bn1=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,kernel_size=k(3),stride=1,padding=k(3)//2)
        self._bn2=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r2=nn.ReLU()
        self._c3=nn.Conv2d(in_channels=bottleneck_depth,out_channels=in_channels,kernel_size=1,stride=1)
        self._bn3=nn.BatchNorm2d(num_features=in_channels)
        self._r3=nn.ReLU()


        self._reduce_rate=4
        self._SE_linear_squeeze=nn.Linear(in_channels,in_channels//self._reduce_rate)
        self._SE_linear_exitation=nn.Linear(in_channels//self._reduce_rate,in_channels)

    def forward(self,x):

        skip=x
        x=self._c1(x)
        x=self._r1(x)
        x=self._bn1(x)
        x=self._c2(x)
        x=self._r2(x)
        x=self._bn2(x)
        x=self._c3(x)
        x=self._bn3(x)
        #SE-block
        global_pooled_x=nn.AdaptiveAvgPool2d(1)(x)
        global_pooled_x=torch.squeeze(global_pooled_x,dim=-1)
        global_pooled_x=torch.squeeze(global_pooled_x,dim=-1)
        squeezed_x=self._SE_linear_squeeze(global_pooled_x)
        squeezed_x=nn.ReLU()(squeezed_x)

        exitated_x=self._SE_linear_exitation(squeezed_x)
        scale_x=nn.Sigmoid()(exitated_x)
        scale_x=scale_x.reshape([-1,self._in_channels,1,1])
        x=x*scale_x


        x=x+skip

        x=self._r3(x)
        return x



class SEResNetBlockShakeShake(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._in_channels=in_channels
        self.out_features=in_channels
        self._branch1=ResNetBasicBlock(in_channels=in_channels)
        self._branch2=ResNetBasicBlock(in_channels=in_channels)

        self._r3=nn.ReLU()


    def forward(self,input_x):

        skip=input_x
        x1=self._branch1(input_x)

        x2=self._branch2(input_x)

        x=ShakeShake.apply(x1,x2,self.training)
        x=x+skip
        x=self._r3(x)
        return x



class SEResNeXtBottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._cardinality=64
        self._in_channels=in_channels
        bottleneck_depth=in_channels//2
        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_depth,
                           kernel_size=1,stride=1)
        self._bn1=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,
                           kernel_size=k(3),stride=1,padding=(k(3))//2,groups=self._cardinality)
        self._bn2=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r2=nn.ReLU()
        self._c3=nn.Conv2d(in_channels=bottleneck_depth,out_channels=in_channels,kernel_size=1,stride=1)
        self._bn3=nn.BatchNorm2d(num_features=in_channels)
        self._r3=nn.ReLU()

        self._reduce_rate=4
        self._SE_linear_squeeze=nn.Linear(in_channels,in_channels//self._reduce_rate)
        self._SE_linear_exitation=nn.Linear(in_channels//self._reduce_rate,in_channels)

    def forward(self,x):

        skip=x
        x=self._c1(x)
        x=self._r1(x)
        x=self._bn1(x)
        x=self._c2(x)
        x=self._r2(x)
        x=self._bn2(x)
        x=self._c3(x)
        x=self._bn3(x)
        #SE-block
        global_pooled_x=nn.AdaptiveAvgPool2d(1)(x)
        global_pooled_x=torch.squeeze(global_pooled_x,dim=-1)
        global_pooled_x=torch.squeeze(global_pooled_x,dim=-1)
        squeezed_x=self._SE_linear_squeeze(global_pooled_x)
        squeezed_x=nn.ReLU()(squeezed_x)

        exitated_x=self._SE_linear_exitation(squeezed_x)
        scale_x=nn.Sigmoid()(exitated_x)
        scale_x=scale_x.reshape([-1,self._in_channels,1,1])
        x=x*scale_x

        _,_,h_skip,w_skip=skip.shape
        _,_,h_x,w_x=x.shape
        if w_skip>w_x:
            x=nn.ReplicationPad2d((w_skip-w_x,0,h_skip-h_x,0))(x)
        else:
            if w_skip<w_x:
                skip=nn.ReplicationPad2d((w_x-w_skip,0,h_x-h_skip,0))(skip)
        x=x+skip

        x=self._r3(x)
        return x
