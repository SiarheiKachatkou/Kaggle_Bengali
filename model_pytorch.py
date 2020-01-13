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
from shake_shake_my import ShakeShake

from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS


def get_augmentations():
    return A.Compose([A.RandomBrightness(p=0.2),
                      A.RandomContrast(p=0.2),
                      A.MotionBlur(p=0.2),
                      A.Cutout(),
                      A.ElasticTransform(alpha=3,sigma=5,alpha_affine=2)],p=0.3)


class ConvBnRelu(torch.nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=3,stride=1):
        super().__init__()
        self._conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=kernel_size//2)
        self._bn=nn.BatchNorm2d(num_features=out_channels)
        self._r=nn.ReLU()

    def forward(self,x):
        x=self._conv(x)
        x=self._bn(x)
        x=self._r(x)
        return x

class ResNetBasicBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=in_channels,out_channels=in_channels,kernel_size=3,stride=1,padding=1)
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
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,kernel_size=3,stride=1,padding=1)
        self._bn2=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
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
    def __init__(self, in_channels,use_shake_shake=False):
        super().__init__()
        self._use_shake_shake=use_shake_shake
        self._in_channels=in_channels
        bottleneck_depth=in_channels//4
        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_depth,kernel_size=1,stride=1)
        self._bn1=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,kernel_size=3,stride=1,padding=1)
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

        if self._use_shake_shake:
            x=ShakeShake.apply(x,skip,self.training)
        else:
            x=x+skip

        x=self._r3(x)
        return x


class SEResNeXtBottleNeckBlock(torch.nn.Module):
    def __init__(self, in_channels, use_shake_shake=True):
        super().__init__()
        self._use_shake_shake=use_shake_shake
        self._cardinality=32
        self._in_channels=in_channels
        bottleneck_depth=in_channels//2
        self._c1=nn.Conv2d(in_channels=in_channels,out_channels=bottleneck_depth,
                           kernel_size=1,stride=1)
        self._bn1=nn.BatchNorm2d(num_features=bottleneck_depth)
        self._r1=nn.ReLU()
        self._c2=nn.Conv2d(in_channels=bottleneck_depth,out_channels=bottleneck_depth,
                           kernel_size=3,stride=1,padding=1,groups=self._cardinality)
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

        if self._use_shake_shake:
            x=ShakeShake.apply(x,skip,self.training)
        else:
            x=x+skip

        x=self._r3(x)
        return x


class Model(ModelBase, torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self._device = torch.device("cuda:0")
        self._layers=[]
        self._print_every_iter=2000
        self._eval_batches=100

        self._classes_list=[]
        #resnet 152,resnet-101,resnet-50
        block_counts_resnet_152=[3,8,36,3]
        block_counts_resnet_101=[3,4,23,3]
        block_counts_resnet_50=[3,4,6,3]
        block_counts_resnet_50_mnist=[3]
        block_counts=block_counts_resnet_50_mnist
        d=4
        self._d=d

        block=SEResNetBottleNeckBlock

        self._blocks=[ConvBnRelu(in_channels=3,out_channels=64//d,stride=2,kernel_size=7),
        ConvBnRelu(in_channels=64//d,out_channels=128//d,stride=2,kernel_size=3),
        ConvBnRelu(in_channels=128//d,out_channels=256//d,stride=1,kernel_size=3)
        ]

        for _ in range(block_counts[0]):
            self._blocks.append(block(in_channels=256//d))

        self._blocks.append(ConvBnRelu(in_channels=256//d,out_channels=512//d,stride=2))
        '''
        for _ in range(block_counts[1]):
            self._blocks.append(block(in_channels=512//d))
        
        self._blocks.append(ConvBnRelu(in_channels=512//d,out_channels=1024//d,stride=2))
        
        for _ in range(block_counts[2]):
            self._blocks.append(block(in_channels=1024//d))
        self._blocks.append(ConvBnRelu(in_channels=1024//d,out_channels=2048//d,stride=2))
        for _ in range(block_counts[3]):
            self._blocks.append(block(in_channels=2048//d))
        '''

        for i,b in enumerate(self._blocks):
            setattr(self,'_block_{}'.format(i),b)


    def forward(self,x):

        for b in self._blocks:
            x=b(x)

        x=torch.mean(x,dim=-1)
        x=torch.mean(x,dim=-1)
        x=torch.flatten(x,1)
        outputs=[]
        for idx,c in enumerate(self._classes_list):
            head=getattr(self,'_head_{}'.format(idx))
            outputs.append(head(x))
        return outputs


    def compile(self,classes_list,**kwargs):
        self._classes_list=classes_list

        in_features=512//self._d
        for idx,c in enumerate(classes_list):
            setattr(self,'_head_{}'.format(idx),torch.nn.Linear(in_features,c))

    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, **kwargs):

        self.to(self._device)

        aug=get_augmentations()
        def aug_fn(img):
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
        optimizer=optim.Adam(self.parameters(),lr=LR)
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)

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
                    scheduler.step(1-val_score)
                    self.train()


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







