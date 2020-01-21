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
from typing import List
from torch.nn.parameter import Parameter
import math
from torch.nn import init
import pretrainedmodels
from torch.nn import Sequential
from cosine_scheduler import CosineScheduler
from transform import affine_image


from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS


def get_augmentations():
    return A.Compose([A.RandomBrightness(p=0.2),
                      A.RandomContrast(p=0.2),
                      A.MotionBlur(p=0.2),
                      A.Cutout(),
                      A.ElasticTransform(alpha=3,sigma=5,alpha_affine=2)],p=0.3)

def residual_add(lhs, rhs):
    lhs_ch, rhs_ch = lhs.shape[1], rhs.shape[1]
    if lhs_ch < rhs_ch:
        out = lhs + rhs[:, :lhs_ch]
    elif lhs_ch > rhs_ch:
        out = torch.cat([lhs[:, :rhs_ch] + rhs, lhs[:, rhs_ch:]], dim=1)
    else:
        out = lhs + rhs
    return out

class LazyLoadModule(nn.Module):
    """Lazy buffer/parameter loading using load_state_dict_pre_hook

    Define all buffer/parameter in `_lazy_buffer_keys`/`_lazy_parameter_keys` and
    save buffer with `register_buffer`/`register_parameter`
    method, which can be outside of __init__ method.
    Then this module can load any shape of Tensor during de-serializing.

    Note that default value of lazy buffer is torch.Tensor([]), while lazy parameter is None.
    """
    _lazy_buffer_keys = []     # It needs to be override to register lazy buffer
    _lazy_parameter_keys = []  # It needs to be override to register lazy parameter

    def __init__(self):
        super(LazyLoadModule, self).__init__()
        for k in self._lazy_buffer_keys:
            self.register_buffer(k, torch.tensor([]))
        for k in self._lazy_parameter_keys:
            self.register_parameter(k, None)
        self._register_load_state_dict_pre_hook(self._hook)

    def _hook(self, state_dict, prefix, local_metadata, strict, missing_keys,
             unexpected_keys, error_msgs):
        for key in self._lazy_buffer_keys:
            self.register_buffer(key, state_dict[prefix + key])

        for key in self._lazy_parameter_keys:
            self.register_parameter(key, Parameter(state_dict[prefix + key]))

class LazyLinear(LazyLoadModule):
    """Linear module with lazy input inference

    `in_features` can be `None`, and it is determined at the first time of forward step dynamically.
    """

    __constants__ = ['bias', 'in_features', 'out_features']
    _lazy_parameter_keys = ['weight']

    def __init__(self, in_features, out_features, bias=True):
        super(LazyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        if in_features is not None:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
            self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.weight is None:
            self.in_features = input.shape[-1]
            self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
            self.reset_parameters()

            # Need to send lazy defined parameter to device...
            self.to(input.device)
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class LinearBlock(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 use_bn=True, activation=F.relu, dropout_ratio=-1, residual=False,):
        super(LinearBlock, self).__init__()
        if in_features is None:
            self.linear = LazyLinear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        if use_bn:
            self.bn = nn.BatchNorm1d(out_features)
        if dropout_ratio > 0.:
            self.dropout = nn.Dropout(p=dropout_ratio)
        else:
            self.dropout = None
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        h = self.linear(x)
        if self.use_bn:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.residual:
            h = residual_add(h, x)
        if self.dropout_ratio > 0:
            h = self.dropout(h)
        return h



class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()

        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        activation = F.leaky_relu
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = LinearBlock(inch, hdim, use_bn=use_bn, activation=activation, residual=False)
        lin2 = LinearBlock(hdim, out_dim, use_bn=use_bn, activation=None, residual=False)
        self.lin_layers = Sequential(lin1, lin2)

    def forward(self, x):
        h = self.base_model.features(x)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
            # [128, 2048, 4, 4] when input is (128, 128)
            bs, ch, height, width = h.shape
            h = h.view(bs, ch*height*width)
        for layer in self.lin_layers:
            h = layer(h)
        return h

def accuracy(y, t):
    pred_label = torch.argmax(y, dim=1)
    count = pred_label.shape[0]
    correct = (pred_label == t).sum().type(torch.float32)
    acc = correct / count
    return acc


class BengaliClassifier(nn.Module):
    def __init__(self, predictor, n_grapheme=168, n_vowel=11, n_consonant=7):
        super(BengaliClassifier, self).__init__()
        self.n_grapheme = n_grapheme
        self.n_vowel = n_vowel
        self.n_consonant = n_consonant
        self.n_total_class = self.n_grapheme + self.n_vowel + self.n_consonant
        self.predictor = predictor

        self.metrics_keys = [
            'loss', 'loss_grapheme', 'loss_vowel', 'loss_consonant',
            'acc_grapheme', 'acc_vowel', 'acc_consonant']

    def forward(self, x, y=None):
        pred = self.predictor(x)
        if isinstance(pred, tuple):
            assert len(pred) == 3
            preds = pred
        else:
            assert pred.shape[1] == self.n_total_class
            preds = torch.split(pred, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)

        if y is None:
            return preds
        else:
            loss_grapheme = F.cross_entropy(preds[0], y[:, 0])
            loss_vowel = F.cross_entropy(preds[1], y[:, 1])
            loss_consonant = F.cross_entropy(preds[2], y[:, 2])
            loss = loss_grapheme + loss_vowel + loss_consonant
            metrics = {
                'loss': loss.item(),
                'loss_grapheme': loss_grapheme.item(),
                'loss_vowel': loss_vowel.item(),
                'loss_consonant': loss_consonant.item(),
                'acc_grapheme': accuracy(preds[0], y[:, 0]),
                'acc_vowel': accuracy(preds[1], y[:, 1]),
                'acc_consonant': accuracy(preds[2], y[:, 2]),
            }
            return loss, metrics, pred

    def calc(self, data_loader):
        device = next(self.parameters()).device
        self.eval()
        output_list = []
        with torch.no_grad():
            for batch in tqdm(data_loader):
                # TODO: support general preprocessing.
                # If `data` is not `Data` instance, `to` method is not supported!
                batch = batch.to(device)
                pred = self.predictor(batch)
                output_list.append(pred)
        output = torch.cat(output_list, dim=0)
        preds = torch.split(output, [self.n_grapheme, self.n_vowel, self.n_consonant], dim=1)
        return preds

    def predict_proba(self, data_loader):
        preds = self.calc(data_loader)
        return [F.softmax(p, dim=1) for p in preds]

    def predict(self, data_loader):
        preds = self.calc(data_loader)
        pred_labels = [torch.argmax(p, dim=1) for p in preds]
        return pred_labels

def build_predictor(arch, out_dim, model_name=None):
    if arch == 'pretrained':
        predictor = PretrainedCNN(in_channels=1, out_dim=out_dim, model_name=model_name)
    else:
        raise ValueError("[ERROR] Unexpected value arch={}".format(arch))
    return predictor


def build_classifier(arch, load_model_path, n_total, model_name='', device='cuda:0'):
    if isinstance(device, str):
        device = torch.device(device)
    predictor = build_predictor(arch, out_dim=n_total, model_name=model_name)
    print('predictor', type(predictor))
    classifier = BengaliClassifier(predictor)
    if load_model_path:
        predictor.load_state_dict(torch.load(load_model_path))
    else:
        print("[WARNING] Unexpected value load_model_path={}"
              .format(load_model_path))
    classifier.to(device)
    return classifier

class Model(ModelBase, torch.nn.Module):


    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self._device = torch.device("cuda")

        self._print_every_iter=1000
        self._eval_batches=100

        self._classes_list=[]

        self._classifier=build_classifier(arch='pretrained', load_model_path=None, n_total=168+11+7, model_name='se_resnext101_32x4d', device='cuda:0')
        #self._classifier=torch.nn.DataParallel(self._classifier)

    def forward(self,x):
        return self._classifier.forward(x)


    def compile(self,classes_list,**kwargs):
        self._classes_list=classes_list


    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, path_to_file, **kwargs):

        self.to(self._device)

        aug=get_augmentations()
        def aug_fn(img):
            img=affine_image(img)
            return img
            #return aug(image=img)['image']

        train_dataset_aug=BengaliDataset(train_images,labels=train_labels,transform_fn=aug_fn)
        train_dataset=BengaliDataset(train_images,labels=train_labels)
        val_dataset=BengaliDataset(val_images,labels=val_labels)

        train_dataloader=DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

        train_val_dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

        val_dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)


        loss_fn=nn.CrossEntropyLoss()
        #optimizer=optim.Adam(self.parameters(),lr=LR)
        optimizer=optim.Adam(self._classifier.predictor.lin_layers.parameters(),lr=LR)
        iter_per_epochs=140000//BATCH_SIZE
        scheduler = CosineScheduler(optimizer, period_initial=iter_per_epochs//2, period_mult=2, lr_initial=0.1, period_warmup_percent=0.1,lr_reduction=0.5)

        for epoch in tqdm(range(EPOCHS)):
            for i, data in enumerate(train_dataloader):

                images,labels=data['image'],data['label']

                images=images.to(self._device,dtype=torch.float)
                labels=labels.to(self._device)

                optimizer.zero_grad()

                heads_outputs = self.__call__(images)

                loss=0
                for idx in range(len(self._classes_list)):
                    loss+=loss_fn(heads_outputs[idx],labels[:,idx])

                loss.backward()

                optimizer.step()

                if i%self._print_every_iter==0:

                    self.eval()
                    with torch.no_grad():
                        train_score=self._eval(train_val_dataloader)
                        val_score=self._eval(val_dataloader)
                        print('loss={} train_score={} val_score={}'.format(loss.item(),train_score,val_score))
                        print('lr={}'.format(scheduler.get_lr()))
                    self.train()

                scheduler.step()

            self.save(path_to_file)


    def _eval(self,dataloader):

        labels_batches=[]
        pred_batches=[]
        for i,data in enumerate(dataloader):
            images,labels=data['image'],data['label']

            preds=self._predict_on_tensor(images)
            labels_batches.append(labels)
            pred_batches.append(preds)

            if i>self._eval_batches:
                break

        preds=np.concatenate(pred_batches,axis=0)
        labels=np.concatenate(labels_batches,axis=0)
        return calc_score(solution=labels,submission=preds)


    def save(self,path_to_file):
        torch.save(self.state_dict(), path_to_file)

    def load(self,path_to_file,classes_list):
        self.compile(classes_list)
        self.load_state_dict(torch.load(path_to_file))
        self.to(self._device)

    def predict(self, images):

        assert isinstance(images,np.ndarray), print('images must be np.array in channel last format')

        self.to(self._device)

        dataset=BengaliDataset(images,labels=None)

        dataloader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

        predicted_labels=[]
        for batch in dataloader:
            inputs=batch['image']
            labels=self._predict_on_tensor(inputs)

            predicted_labels.append(labels)

        return np.concatenate(predicted_labels,axis=0)

    def _predict_on_tensor(self,inputs):

        self.eval()

        def _argmax(tensor):
            return tensor.data.cpu().numpy().argmax(axis=1).reshape([-1,1])

        inputs=inputs.to(self._device)
        heads = self.__call__(inputs)
        labels=[_argmax(head) for head in heads]
        labels = np.hstack(labels)
        return labels







