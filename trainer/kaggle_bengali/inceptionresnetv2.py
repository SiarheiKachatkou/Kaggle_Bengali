from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

pretrained_settings = {
    'inceptionresnetv2': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1000
        },
        'imagenet+background': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/inceptionresnetv2-520b38e4.pth',
            'input_space': 'RGB',
            'input_size': [3, 299, 299],
            'input_range': [0, 1],
            'mean': [0.5, 0.5, 0.5],
            'std': [0.5, 0.5, 0.5],
            'num_classes': 1001
        }
    }
}

def mult_filters(filters,width):
    return max(3,int(filters*width))

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
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_5b(nn.Module):

    def __init__(self,in_features, width):
        super(Mixed_5b, self).__init__()

        self._width=width

        def _m(filters):
            return mult_filters(filters,self._width)

        self.branch0 = BasicConv2d(in_features, _m(96), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(48), kernel_size=1, stride=1),
            BasicConv2d(_m(48), _m(64), kernel_size=5, stride=1, padding=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_features, _m(64), kernel_size=1, stride=1),
            BasicConv2d(_m(64), _m(96), kernel_size=3, stride=1, padding=1),
            BasicConv2d(_m(96), _m(96), kernel_size=3, stride=1, padding=1)
        )

        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(in_features, _m(64), kernel_size=1, stride=1)
        )

        self.out_features=_m(96)+_m(64)+_m(96)+_m(64)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block35(nn.Module):

    def __init__(self, in_features, width, scale=1.0):
        super(Block35, self).__init__()

        self._width=width

        def _m(filters):
            return mult_filters(filters,self._width)

        self.scale = scale

        self.branch0 = BasicConv2d(in_features, _m(32), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(32), kernel_size=1, stride=1),
            BasicConv2d(_m(32), _m(32), kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_features, _m(32), kernel_size=1, stride=1),
            BasicConv2d(_m(32), _m(48), kernel_size=3, stride=1, padding=1),
            BasicConv2d(_m(48), _m(64), kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(_m(32)+_m(32)+_m(64), in_features, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

        self.out_features=in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self,in_features,width):
        super(Mixed_6a, self).__init__()

        self._width=width

        def _m(filters):
            return mult_filters(filters,self._width)

        self.branch0 = BasicConv2d(in_features, _m(384), kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(256), kernel_size=1, stride=1),
            BasicConv2d(_m(256), _m(256), kernel_size=3, stride=1, padding=1),
            BasicConv2d(_m(256), _m(384), kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

        self.out_features=_m(384)+_m(384)+in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Block17(nn.Module):

    def __init__(self, in_features, width, scale=1.0):
        super(Block17, self).__init__()

        self.scale = scale

        self._width=width

        def _m(filters):
            return mult_filters(filters,self._width)

        self.branch0 = BasicConv2d(in_features, _m(192), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(128), kernel_size=1, stride=1),
            BasicConv2d(_m(128), _m(160), kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(_m(160), _m(192), kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(_m(192)+_m(192), in_features, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

        self.out_features=in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Mixed_7a(nn.Module):

    def __init__(self,in_features,width):
        super(Mixed_7a, self).__init__()

        self._width=width
        def _m(filters):
            return mult_filters(filters,self._width)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_features, _m(256), kernel_size=1, stride=1),
            BasicConv2d(_m(256), _m(384), kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(256), kernel_size=1, stride=1),
            BasicConv2d(_m(256), _m(288), kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(in_features, _m(256), kernel_size=1, stride=1),
            BasicConv2d(_m(256), _m(288), kernel_size=3, stride=1, padding=1),
            BasicConv2d(_m(288), _m(320), kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

        self.out_features=_m(384)+_m(288)+_m(320)+in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Block8(nn.Module):

    def __init__(self, width, in_features, scale=1.0, noReLU=False):
        super(Block8, self).__init__()

        self._width=width
        def _m(filters):
            return mult_filters(filters,self._width)


        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(in_features, _m(192), kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(in_features, _m(192), kernel_size=1, stride=1),
            BasicConv2d(_m(192), _m(224), kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(_m(224), _m(256), kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(_m(256)+_m(192), in_features, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

        self.out_features=in_features

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class InceptionResNetV2(nn.Module):

    def __init__(self, repeats=(10,20,9), width=1, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        self._width=width
        def _m(filters):
            return mult_filters(filters,self._width)

        # Modules
        self.conv2d_1a = BasicConv2d(3, _m(32), kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(_m(32), _m(32), kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(_m(32), _m(64), kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(_m(64), _m(80), kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(_m(80), _m(192), kernel_size=3, stride=1)
        self.maxpool_5a = nn.MaxPool2d(3, stride=2)
        self.mixed_5b = Mixed_5b(width=width,in_features=_m(192))
        self.repeat = nn.Sequential(*[Block35(scale=0.17,width=width,in_features=self.mixed_5b.out_features) for _ in range(repeats[0])])
        self.mixed_6a = Mixed_6a(width=width,in_features=self.mixed_5b.out_features)
        self.repeat_1 = nn.Sequential(*[Block17(scale=0.1,width=width,in_features=self.mixed_6a.out_features) for _ in range(repeats[1])])
        self.mixed_7a = Mixed_7a(width=width,in_features=self.mixed_6a.out_features)
        self.repeat_2 = nn.Sequential(*[Block8(scale=0.2,width=width,in_features=self.mixed_7a.out_features) for _ in range(repeats[1])])
        self.block8 = Block8(noReLU=True,width=width,in_features=self.mixed_7a.out_features)
        self.conv2d_7b = BasicConv2d(self.block8.out_features, _m(1536), kernel_size=1, stride=1)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.last_linear = nn.Linear(_m(1536), num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.maxpool_5a(x)
        x = self.mixed_5b(x)
        x = self.repeat(x)
        x = self.mixed_6a(x)
        x = self.repeat_1(x)
        x = self.mixed_7a(x)
        x = self.repeat_2(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x

def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    if pretrained:
        settings = pretrained_settings['inceptionresnetv2'][pretrained]
        assert num_classes == settings['num_classes'], \
            "num_classes should be {}, but is {}".format(settings['num_classes'], num_classes)

        # both 'imagenet'&'imagenet+background' are loaded from same parameters
        model = InceptionResNetV2(num_classes=1001)
        model.load_state_dict(model_zoo.load_url(settings['url']))

        if pretrained == 'imagenet':
            new_last_linear = nn.Linear(1536, 1000)
            new_last_linear.weight.data = model.last_linear.weight.data[1:]
            new_last_linear.bias.data = model.last_linear.bias.data[1:]
            model.last_linear = new_last_linear

        model.input_space = settings['input_space']
        model.input_size = settings['input_size']
        model.input_range = settings['input_range']

        model.mean = settings['mean']
        model.std = settings['std']
    else:
        model = InceptionResNetV2(num_classes=num_classes)
    return model

'''
TEST
Run this code with:
```
cd $HOME/pretrained-models.pytorch
python -m pretrainedmodels.inceptionresnetv2
```
'''
if __name__ == '__main__':

    assert inceptionresnetv2(num_classes=10, pretrained=None)
    print('success')
    assert inceptionresnetv2(num_classes=1000, pretrained='imagenet')
    print('success')
    assert inceptionresnetv2(num_classes=1001, pretrained='imagenet+background')
    print('success')

    # fail
    assert inceptionresnetv2(num_classes=1001, pretrained='imagenet')
