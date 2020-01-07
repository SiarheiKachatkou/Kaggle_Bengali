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
from shake_shake.shake_resnet import ShakeResNet

from consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS


def get_augmentations():
    return A.Compose([A.RandomBrightness(p=0.2),
                      A.RandomContrast(p=0.2),
                      A.MotionBlur(p=0.2),
                      A.CLAHE(clip_limit=1,p=0.1),
                      A.CLAHE(clip_limit=4,p=0.1),
                      A.Cutout(),
                      A.RandomSnow(p=0.2),
                      A.RandomRain(p=0.2),
                      A.RandomFog(p=0.2),
                      A.ElasticTransform(alpha=3,sigma=5,alpha_affine=2)],p=0.3)

class Model(ModelBase, torch.nn.Module):

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self._device = torch.device("cuda:0")

        self._print_every_iter=2000
        self._eval_batches=100

        self._classes_list=[]

        self._graph=ShakeResNet(depth=26,w_base=64,label=168)
        self._vowel=ShakeResNet(depth=26,w_base=64,label=11)
        self._conso=ShakeResNet(depth=26,w_base=64,label=7)



    def forward(self,x):

        fc_graph=self._graph(x)
        fc_vowel = self._vowel(x)
        fc_conso=self._conso(x)

        return fc_graph, fc_vowel, fc_conso



    def compile(self,classes_list,**kwargs):
        self._classes_list=classes_list

    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, **kwargs):

        self.to(self._device)

        aug=get_augmentations()
        def aug_fn(img):
            #return img
            return aug(image=img)['image']

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
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)

        for epoch in tqdm(range(EPOCHS)):
            for i, data in enumerate(train_dataloader):

                images,labels=data['image'],data['label']

                images=images.to(self._device,dtype=torch.float)
                labels=labels.to(self._device)

                optimizer.zero_grad()

                fc_graph, fc_vowel, fc_conso = self.__call__(images)

                loss=2*loss_fn(fc_graph,labels[:,0])+loss_fn(fc_vowel,labels[:,1])+loss_fn(fc_conso,labels[:,2])

                loss.backward()

                optimizer.step()

                if i%self._print_every_iter==0:

                    train_score=self._eval(train_val_dataloader)
                    val_score=self._eval(val_dataloader)
                    print('loss={} train_score={} val_score={}'.format(loss.item(),train_score,val_score))
                    scheduler.step(1-val_score)


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
        graph,vowel,conso = self.__call__(inputs)

        graph_labels=_argmax(graph)
        vowel_labels=_argmax(vowel)
        conso_labels=_argmax(conso)
        labels = np.hstack([graph_labels,vowel_labels,conso_labels])
        return labels







