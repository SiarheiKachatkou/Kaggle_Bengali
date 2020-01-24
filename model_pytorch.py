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
import cv2
from typing import List
from torch.nn.parameter import Parameter
import math
from torch.nn import init
import pretrainedmodels
from torch.nn import Sequential
from cosine_scheduler import CosineScheduler
from loss import calc_classes_weights, RecallScore
from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS, AUGM_PROB, DROPOUT_P, LOSS_WEIGHTS

mode='FP32'
opt_level='O3'

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    print(" apex not avalaible Please install apex from https://www.github.com/nvidia/apex to run this example.")


def get_augmentations():
    return A.OneOf([
                      A.RandomContrast(limit=(0.8,1.2),p=0.2),
                      A.MotionBlur(blur_limit=15,p=0.2),
                      A.Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0,p=0.8),
                      A.Cutout(num_holes=16, max_h_size=4, max_w_size=4, fill_value=255,p=0.8),
                      A.Cutout(num_holes=3, max_h_size=20, max_w_size=20, fill_value=0,p=0.8),
                      A.Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=255,p=0.8),
                      A.ShiftScaleRotate(shift_limit=0.06,scale_limit=0.1,rotate_limit=15,border_mode=cv2.BORDER_CONSTANT,value=255,p=0.8),
                      A.ElasticTransform(alpha=30,sigma=5,alpha_affine=10,border_mode=cv2.BORDER_CONSTANT,value=255,p=1.0),
                      A.ElasticTransform(alpha=60,sigma=15,alpha_affine=20,border_mode=cv2.BORDER_CONSTANT,value=255,p=1.0),
                      ],p=AUGM_PROB)



class PretrainedCNN(nn.Module):
    def __init__(self, model_name='se_resnext101_32x4d',
                 in_channels=1, out_dim=10, use_bn=True,
                 pretrained='imagenet'):
        super(PretrainedCNN, self).__init__()

        self.base_model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        self.do_pooling = True
        if self.do_pooling:
            inch = self.base_model.last_linear.in_features
        else:
            inch = None
        hdim = 512
        lin1 = nn.Linear(in_features=inch, out_features=hdim)
        r1=nn.ReLU()
        d1=nn.Dropout2d(p=DROPOUT_P)
        lin2 = nn.Linear(in_features=hdim, out_features=out_dim)
        d2=nn.Dropout2d(p=DROPOUT_P)

        self.lin_layers = Sequential(d1,lin1, r1, d2, lin2)

    def forward(self, x):
        h = self.base_model.features(x)

        if self.do_pooling:
            h = torch.sum(h, dim=(-1, -2))
        else:
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


def build_classifier(n_total, model_name='se_resnext101_32x4d', device='cuda:0'):
    predictor = PretrainedCNN(in_channels=N_CHANNELS, out_dim=n_total, model_name=model_name)
    classifier = BengaliClassifier(predictor)
    classifier.to(device)
    return classifier

class Model(ModelBase, torch.nn.Module):

    def _m(self,channels):
        return int(self._d*channels)

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self._device = torch.device("cuda:0")
        self._print_every_iter=2000
        self._eval_batches=100

        self._classes_list=[]

        self._classifier=build_classifier(n_total=168+11+7, model_name='se_resnext101_32x4d', device='cuda:0')

    def forward(self,x):

        return self._classifier(x)


    def compile(self,classes_list,**kwargs):
        self._classes_list=classes_list

    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, **kwargs):

        self.to(self._device)

        aug=get_augmentations()
        def aug_fn(img):
            return aug(image=img)['image']


        classes_weights=calc_classes_weights(train_labels,self._classes_list)

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


        loss_fns=[RecallScore(class_weights) for class_weights in classes_weights]
        optimizer=optim.Adam(self.parameters(),lr=LR)
        iter_per_epochs=140000//BATCH_SIZE
        scheduler = CosineScheduler(optimizer, period_initial=iter_per_epochs//2, period_mult=2, lr_initial=LR, period_warmup_percent=0.1,lr_reduction=0.5)

        if mode == 'FP16':
            self._classifier = network_to_half(self._classifier)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=128)
        elif mode == 'amp':
            self._classifier, optimizer = amp.initialize(self._classifier, optimizer, opt_level=opt_level)


        for epoch in tqdm(range(EPOCHS)):
            for i, data in enumerate(train_dataloader):

                images,labels=data['image'],data['label']

                images=images.to(self._device,dtype=torch.float)
                labels=labels.to(self._device)

                optimizer.zero_grad()

                heads_outputs = self.__call__(images)

                loss=0
                for idx in range(len(self._classes_list)):
                    this_loss=LOSS_WEIGHTS[idx]*loss_fns[idx](heads_outputs[idx],labels[:,idx])
                    loss+=this_loss


                if mode == 'FP32':
                    loss.backward()
                    optimizer.step()
                elif mode == 'FP16':
                    optimizer.backward(loss)
                else:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()


                if i%self._print_every_iter==0:

                    self.eval()
                    with torch.no_grad():
                        train_score=self._eval(train_val_dataloader)
                        val_score=self._eval(val_dataloader)
                        print('loss={} train_score={} val_score={}'.format(loss.item(),train_score,val_score))
                        print('lr={}'.format(scheduler.get_lr()))
                    self.train()

                scheduler.step()



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

        def _argmax(tensor):
            return tensor.data.cpu().numpy().argmax(axis=1).reshape([-1,1])

        inputs=inputs.to(self._device)
        heads = self.__call__(inputs)
        labels=[_argmax(head) for head in heads]
        labels = np.hstack(labels)
        return labels







