import torch
import tensorflow as tf
from datetime import datetime
import torchvision
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import albumentations as A
import pretrainedmodels
from torch.utils.data import WeightedRandomSampler
from .model_base import ModelBase
from .score import calc_score
from .dataset_pytorch import BengaliDataset
from .shake_shake_my import ShakeShake
from .consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS, AUGM_PROB,FAST_PROTO_SCALE, \
    DROPOUT_P, LOSS_WEIGHTS, LR_SCHEDULER_PATINCE,USE_FREQ_SAMPLING,CLASSES_LIST
from .loss import calc_classes_weights, RecallScore
from .save_to_maybe_gs import save

def k(kernel_size):
    return kernel_size
    return max(1,round(kernel_size/FAST_PROTO_SCALE))


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

class BackBone(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone=pretrainedmodels.resnet152(pretrained=None)
        in_features=self._backbone.last_linear.in_features
        self._last_linear=nn.Linear(in_features=in_features,out_features=np.sum(CLASSES_LIST),bias=True)

    def forward(self, x):
        x=self._backbone.features(x)
        x=torch.nn.AdaptiveAvgPool2d(1)(x)
        x=torch.squeeze(x,dim=-1)
        x=torch.squeeze(x,dim=-1)
        x=self._last_linear(x)
        return x




class Model(ModelBase, torch.nn.Module):

    def _m(self,channels):
        return int(self._d*channels)

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)

        self._device = torch.device("cuda")
        self._layers=[]
        self._print_every_iter=2000
        self._eval_batches=100

        self._classes_list=[]

        self._backbone=BackBone()
        #self._backbone=nn.DataParallel(self._backbone)

    def forward(self,x):
        x=self._backbone(x)
        outputs=torch.split(x,self._classes_list,dim=1)
        return outputs

    def compile(self,classes_list,**kwargs):
        self._classes_list=classes_list


    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, path_to_model_save, **kwargs):

        #torch.distributed.init_process_group(backend='nccl')

        self.to(self._device)

        tf.logging.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

        aug=get_augmentations()
        def aug_fn(img):
            if FAST_PROTO_SCALE!=1:
                img=cv2.resize(img,(round(IMG_W/FAST_PROTO_SCALE),round(IMG_H/FAST_PROTO_SCALE)))
            return aug(image=img)['image']

        train_dataset_aug=BengaliDataset(train_images,labels=train_labels,transform_fn=aug_fn)
        train_dataset=BengaliDataset(train_images,labels=train_labels)
        val_dataset=BengaliDataset(val_images,labels=val_labels)

        classes_weights=calc_classes_weights(train_labels,self._classes_list)

        train_sampler = None#torch.utils.data.distributed.DistributedSampler(train_dataset_aug)

        train_dataloader=DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=False, sampler=train_sampler,
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
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATINCE, verbose=True,
                                                             threshold=0.0001, threshold_mode='abs',
                                                             cooldown=0, min_lr=1e-6, eps=1e-08)
        start_time=datetime.now()
        for epoch in tqdm(range(EPOCHS)):
            tf.logging.info('epoch {}/{}'.format(epoch+1,EPOCHS))
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

                loss.backward()

                optimizer.step()

                if i%self._print_every_iter==0:

                    self.eval()
                    with torch.no_grad():
                        train_score=self._eval(train_val_dataloader)
                        val_score=self._eval(val_dataloader)
                        tf.logging.info('loss={} train_score={} val_score={}'.format(loss.item(),train_score,val_score))
                        time=(datetime.now()-start_time).seconds
                        tf.logging.info('iter/secs={}   lr={}'.format(self._print_every_iter/time,optimizer.param_groups[0]['lr']))
                        start_time=datetime.now()

                    self.train()
                    scheduler.step(1-val_score)
            save(self.save,path_to_model_save)



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







