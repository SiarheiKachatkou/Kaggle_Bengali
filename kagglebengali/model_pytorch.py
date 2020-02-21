import torch
from datetime import datetime
import torchvision
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import albumentations as A
import pretrainedmodels
from .radam import RAdam
from .model_base import ModelBase
from .score import calc_score
from .dataset_pytorch import BengaliDataset
from .consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS, AUGM_PROB,\
    DROPOUT_P, LOSS_WEIGHTS, LR_SCHEDULER_PATINCE,CLASSES_LIST, LOG_FILENAME, FEATURES_AREA, EVAL_EVERY_STEPS,EVAL_BATCHES,CLASSES_LIST_ORIG,TRAIN_ONLY_LAST_FULLY_CONNECTED
from .loss import calc_classes_weights, RecallScore
from .save_to_maybe_gs import save
from .local_logging import get_logger
from .resnet import resnet152

logger=get_logger(__name__)


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

        in_features=self._backbone.last_linear.in_features*FEATURES_AREA
        self.fully_connected=nn.Linear(in_features=in_features,out_features=np.sum(CLASSES_LIST),bias=True)

    def forward(self, x):
        x=self._backbone.features(x)
        x=x.view(BATCH_SIZE,-1)
        x=self.fully_connected(x)
        return x




class Model(ModelBase, torch.nn.Module):

    def _m(self,channels):
        return int(self._d*channels)

    def __init__(self):
        torch.nn.Module.__init__(self)
        ModelBase.__init__(self)
        self._train_only_last_fully_connected=TRAIN_ONLY_LAST_FULLY_CONNECTED
        self._device = torch.device("cuda")
        self._print_every_iter=EVAL_EVERY_STEPS
        self._eval_batches=EVAL_BATCHES

        self._classes_list=CLASSES_LIST

        self._backbone=BackBone()
        #self._backbone=nn.DataParallel(self._backbone)

        #optimizer=optim.Adam(self.parameters(),lr=LR)
        self._optimizer=RAdam(self.parameters(),lr=LR)

    def forward(self,x):
        x=self._backbone(x)
        outputs=torch.split(x,self._classes_list,dim=1)
        return outputs

    def compile(self,classes_list,**kwargs):
        pass


    def fit(self,train_images,train_labels, val_images, val_labels, batch_size,epochs, path_to_model_save, **kwargs):

        self.to(self._device)

        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

        aug=get_augmentations()
        def aug_fn(img):
            return aug(image=img)['image']

        train_dataset_aug=BengaliDataset(train_images,labels=train_labels,transform_fn=aug_fn)
        train_dataset=BengaliDataset(train_images,labels=train_labels)
        val_dataset=BengaliDataset(val_images,labels=val_labels)

        train_sampler = None

        train_dataloader=DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True, sampler=train_sampler,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        train_val_dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        val_dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        if self._train_only_last_fully_connected:
            self._classes_list=CLASSES_LIST_ORIG
            in_features=self._backbone.fully_connected.in_features
            self._backbone.fully_connected=nn.Linear(in_features=in_features,out_features=np.sum(self._classes_list),bias=True)
            self.to(self._device)

        self._optimizer=RAdam(self._backbone.fully_connected.parameters(),lr=LR)

        loss_fns=[RecallScore(None) for _ in range(3)]

        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATINCE, verbose=True,
                                                             threshold=0.0001, threshold_mode='abs',
                                                             cooldown=0, min_lr=1e-6, eps=1e-08)

        start_time=datetime.now()
        global_step=0
        best_eval_metric=0
        for epoch in tqdm(range(EPOCHS)):
            logger.info('epoch {}/{}'.format(epoch+1,EPOCHS))
            for i, data in enumerate(train_dataloader):

                global_step+=1

                images,labels=data['image'],data['label']

                images=images.to(self._device,dtype=torch.float)
                labels=labels.to(self._device)

                self._optimizer.zero_grad()

                heads_outputs = self.__call__(images)

                loss=0
                for idx in range(len(self._classes_list)):
                    this_loss=LOSS_WEIGHTS[idx]*loss_fns[idx](heads_outputs[idx],labels[:,idx])
                    loss+=this_loss

                loss.backward()

                self._optimizer.step()

                if global_step%self._print_every_iter==0 or global_step==1:

                    self.eval()
                    with torch.no_grad():
                        train_score=self._eval(train_val_dataloader)
                        val_score=self._eval(val_dataloader)
                        logger.info('loss={} train_score={} val_score={}'.format(loss.item(),train_score,val_score))
                        time=(datetime.now()-start_time).seconds

                        logger.info('iter/secs={}   lr={}'.format(global_step/time,self._optimizer.param_groups[0]['lr']))

                    self.train()
                    scheduler.step(1-val_score)

                    def copy_log_file(dst_path):
                        shutil.copyfile(LOG_FILENAME,dst_path)

                    save(copy_log_file, path_to_model_save+'_log.txt')

                    if val_score>best_eval_metric:
                        best_eval_metric=val_score
                        save(self.save,path_to_model_save)
                        logger.info('new best model saved')




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
        state_dict={'model':self.state_dict(),'optimizer':self._optimizer.state_dict()}
        torch.save(state_dict, path_to_file)


    def load(self,path_to_file,classes_list):
        self.compile(classes_list)
        state_dict=torch.load(path_to_file)
        self.load_state_dict(state_dict['model'])
        self._optimizer.load_state_dict(state_dict['optimizer'])
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







