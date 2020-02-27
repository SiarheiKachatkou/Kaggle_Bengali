from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionV4', 'inceptionv4']

pretrained_settings = {
    'inceptionv4': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionv4-8e4777a0.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}


class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False) # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001, # value found in tensorflow
                                 momentum=0.1, # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3a(nn.Module):

    def __init__(self,m,in_features):
        super(Mixed_3a, self).__init__()
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.conv = BasicConv2d(in_features, m(96), kernel_size=3, stride=2)
        self.out_features=in_features+m(96)

    def forward(self, x):
        x0 = self.maxpool(x)
        x1 = self.conv(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_4a(nn.Module):

    def __init__(self,m,in_features):
        super(Mixed_4a, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_features, m(64), kernel_size=1, stride=1),
            BasicConv2d(m(64), m(96), kernel_size=3, stride=1)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, m(64), kernel_size=1, stride=1),
            BasicConv2d(m(64), m(64), kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(m(64), m(64), kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(m(64), m(96), kernel_size=(3,3), stride=1)
        )

        self.out_features=m(96)+m(96)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out


class Mixed_5a(nn.Module):

    def __init__(self,m,in_features):
        super(Mixed_5a, self).__init__()
        self.conv = BasicConv2d(in_features, m(192), kernel_size=3, stride=2)
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.out_features=m(192)+in_features

    def forward(self, x):
        x0 = self.conv(x)
        x1 = self.maxpool(x)
        out = torch.cat((x0, x1), 1)
        return out


class Inception_A(nn.Module):

    def __init__(self,m,in_features):
        super(Inception_A, self).__init__()
        self.branch0 = BasicConv2d(in_features, m(96), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features,m(64), kernel_size=1, stride=1),
            BasicConv2d(m(64), m(96), kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_features,m(64), kernel_size=1, stride=1),
            BasicConv2d(m(64), m(96), kernel_size=3, stride=1, padding=1),
            BasicConv2d(m(96), m(96), kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_features, m(96), kernel_size=1, stride=1)
        )

        self.out_features=m(96)*4

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_A(nn.Module):

    def __init__(self,m,in_features):
        super(Reduction_A, self).__init__()
        self.branch0 = BasicConv2d(in_features, m(384), kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, m(192), kernel_size=1, stride=1),
            BasicConv2d(m(192), m(224), kernel_size=3, stride=1, padding=1),
            BasicConv2d(m(224), m(256), kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        self.out_features=m(384)+m(256)+in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_B(nn.Module):

    def __init__(self,m,in_features):
        super(Inception_B, self).__init__()
        self.branch0 = BasicConv2d(in_features, m(384), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, m(192), kernel_size=1, stride=1),
            BasicConv2d(m(192), m(224), kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(m(224), m(256), kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_features, m(192), kernel_size=1, stride=1),
            BasicConv2d(m(192), m(192), kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(m(192), m(224), kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(m(224), m(224), kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(m(224), m(256), kernel_size=(1,7), stride=1, padding=(0,3))
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_features, m(128), kernel_size=1, stride=1)
        )

        self.out_features=m(384)+m(256)+m(256)+m(128)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Reduction_B(nn.Module):

    def __init__(self,m,in_features):
        super(Reduction_B, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(in_features, m(192), kernel_size=1, stride=1),
            BasicConv2d(m(192), m(192), kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, m(256), kernel_size=1, stride=1),
            BasicConv2d(m(256), m(256), kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(m(256), m(320), kernel_size=(7,1), stride=1, padding=(3,0)),
            BasicConv2d(m(320), m(320), kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        self.out_features=m(192)+m(320)+in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Inception_C(nn.Module):

    def __init__(self,m,in_features):
        super(Inception_C, self).__init__()

        self.branch0 = BasicConv2d(in_features, m(256), kernel_size=1, stride=1)

        self.branch1_0 = BasicConv2d(in_features, m(384), kernel_size=1, stride=1)
        self.branch1_1a = BasicConv2d(m(384), m(256), kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch1_1b = BasicConv2d(m(384), m(256), kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch2_0 = BasicConv2d(in_features, m(384), kernel_size=1, stride=1)
        self.branch2_1 = BasicConv2d(m(384), m(448), kernel_size=(3,1), stride=1, padding=(1,0))
        self.branch2_2 = BasicConv2d(m(448), m(512), kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3a = BasicConv2d(m(512), m(256), kernel_size=(1,3), stride=1, padding=(0,1))
        self.branch2_3b = BasicConv2d(m(512), m(256), kernel_size=(3,1), stride=1, padding=(1,0))

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_features, m(256), kernel_size=1, stride=1)
        )

        self.out_features=m(256)+m(256)+m(256)+m(256)+m(256)+m(256)

    def forward(self, x):
        x0 = self.branch0(x)

        x1_0 = self.branch1_0(x)
        x1_1a = self.branch1_1a(x1_0)
        x1_1b = self.branch1_1b(x1_0)
        x1 = torch.cat((x1_1a, x1_1b), 1)

        x2_0 = self.branch2_0(x)
        x2_1 = self.branch2_1(x2_0)
        x2_2 = self.branch2_2(x2_1)
        x2_3a = self.branch2_3a(x2_2)
        x2_3b = self.branch2_3b(x2_2)
        x2 = torch.cat((x2_3a, x2_3b), 1)

        x3 = self.branch3(x)

        out = torch.cat((x0, x1, x2, x3), 1)
        return out

def mult_filters(filters,width):
    return max(3,int(filters*width))

class InceptionV4(nn.Module):

    def __init__(self, repeats=(10,20,9), width=1,num_classes=1001):
        super(InceptionV4, self).__init__()

        self._width=width
        def _m(filters):
            return mult_filters(filters,self._width)
        # Modules
        self.features = nn.Sequential(
            BasicConv2d(3, _m(32), kernel_size=3, stride=2),
            BasicConv2d(_m(32), _m(32), kernel_size=3, stride=1),
            BasicConv2d(_m(32), _m(64), kernel_size=3, stride=1, padding=1))

        modules=[Mixed_3a,Mixed_4a,Mixed_5a]
        modules.extend([Inception_A]*repeats[0])
        modules.append(Reduction_A)
        modules.extend([Inception_B]*repeats[1])
        modules.append(Reduction_B)
        modules.extend([Inception_C]*repeats[2])

        in_features=_m(64)
        for m in modules:
            module=m(_m,in_features)
            in_features=module.out_features
            self.features.add_module(str(len(self.features)),module)

        self.last_linear = nn.Linear(in_features, num_classes)

    def logits(self, features):
        #Allows image of any size to be processed
        adaptiveAvgPoolWidth = features.shape[2]
        x = F.avg_pool2d(features, kernel_size=adaptiveAvgPoolWidth)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x
