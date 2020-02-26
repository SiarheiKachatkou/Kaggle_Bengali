import torch
from datetime import datetime
import pytorch_lightning as pl
import shutil
import numpy as np
import cv2
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import albumentations as A
from .radam import RAdam
from .score import calc_score
from .dataset_pytorch import BengaliDataset
from .consts import IMG_W,IMG_H,N_CHANNELS, BATCH_SIZE, LR, EPOCHS, AUGM_PROB, \
    DROPOUT_P, LOSS_WEIGHTS, LR_SCHEDULER_PATINCE,CLASSES_LIST, LOG_FILENAME, \
    BACKBONE_FN,BACKBONE_KWARGS
from .loss import calc_classes_weights, RecallScore
from .save_to_maybe_gs import save
from ..local_logging import get_logger
from .resnet import _resnet


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
        self._backbone=BACKBONE_FN(**BACKBONE_KWARGS)

    def forward(self, x):

        x=self._backbone(x)
        return x




class Model(pl.LightningModule):

    def __init__(self,train_images,train_labels, val_images, val_labels,path_to_model_save):
        super(Model,self).__init__()

        self._path_to_model_save=path_to_model_save

        self._classes_list=CLASSES_LIST
        self._backbone=BackBone()

        self._aug=get_augmentations()

        train_dataset_aug=BengaliDataset(train_images,labels=train_labels,transform_fn=self._aug_fn)
        train_dataset=BengaliDataset(train_images,labels=train_labels)
        val_dataset=BengaliDataset(val_images,labels=val_labels)

        self._train_dataloader=DataLoader(train_dataset_aug, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=16, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        self._train_val_dataloader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=16, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        self._val_dataloader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, sampler=None,
           batch_sampler=None, num_workers=16, collate_fn=None,
           pin_memory=False, drop_last=True, timeout=0,
           worker_init_fn=None)

        classes_weights=calc_classes_weights(train_labels,self._classes_list)
        self._loss_fns=[RecallScore(class_weights) for class_weights in classes_weights]

        logger.info("Let's use {} GPUs!".format(torch.cuda.device_count()))

        self._start_time=datetime.now()
        self._global_step=0

    def _aug_fn(self,img):
            return self._aug(image=img)['image']

    def forward(self,x):
        x=self._backbone(x)
        outputs=torch.split(x,self._classes_list,dim=1)
        return outputs

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return self._train_dataloader

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return self._val_dataloader

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return self._val_dataloader

    def _calc_loss(self,heads_outputs,labels):
        loss=0
        for idx in range(len(self._classes_list)):
            this_loss=LOSS_WEIGHTS[idx]*self._loss_fns[idx](heads_outputs[idx],labels[:,idx])
            loss+=this_loss
        return loss

    def training_step(self,batch,batch_nb):

        self._global_step+=1

        images,labels=batch['image'],batch['label']

        heads_outputs = self.forward(images)

        loss=self._calc_loss(heads_outputs,labels)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}


    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        images,labels=batch['image'],batch['label']
        predicted_labels, predicted_logits = self._predict_on_tensor(images)
        return {'predicted_labels': predicted_labels,'predicted_logits': predicted_logits,'true_labels':labels}

    def finalize_log(self):
        def copy_log_file(dst_path):
                shutil.copyfile(LOG_FILENAME,dst_path)

        save(copy_log_file, self._path_to_model_save+'_log.txt')

    def validation_end(self, outputs):
        # OPTIONAL
        pred_batches=[x['predicted_labels'] for x in outputs]
        labels_batches=[x['true_labels'] for x in outputs]
        logits_batches=[x['predicted_logits'] for x in outputs]
        loss=[self._calc_loss(logit,label).item() for logit,label in zip(logits_batches,labels_batches)]
        loss=np.mean(loss)
        preds=np.concatenate(pred_batches,axis=0)

        labels_batches=[labels_batch.cpu().numpy() for labels_batch in labels_batches]
        labels=np.concatenate(labels_batches,axis=0)

        val_score=calc_score(solution=labels,submission=preds)
        self.scheduler.step(val_score)

        logger.info('val_loss={} val_score={}'.format(loss,val_score))
        time=(datetime.now()-self._start_time).seconds
        eps=1e-3
        logger.info('iter/secs={}   lr={}'.format(self._global_step/(time+eps),self.optimizer.param_groups[0]['lr']))


        self.finalize_log()
        save(self.save,self._path_to_model_save)

        return {'val_score': val_score,'val_loss':loss}


    def configure_optimizers(self):
        # REQUIRED
        self.optimizer=RAdam(self.parameters(),lr=LR)
        self.scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=LR_SCHEDULER_PATINCE, verbose=True,
                                                             threshold=0.0001, threshold_mode='abs',
                                                             cooldown=0, min_lr=1e-6, eps=1e-08)
        return self.optimizer


    def save(self,path_to_file):
        torch.save(self.state_dict(), path_to_file)

    def load(self,path_to_file,classes_list):
        self.compile(classes_list)
        self.load_state_dict(torch.load(path_to_file))
        self.to(self._device)

    def predict(self, images):

        assert isinstance(images,np.ndarray), print('images must be np.array in channel last format')

        dataset=BengaliDataset(images,labels=None)

        dataloader=DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None)

        predicted_labels=[]
        for batch in dataloader:
            inputs=batch['image']
            inputs=inputs.cuda()
            labels,_=self._predict_on_tensor(inputs)

            predicted_labels.append(labels)

        return np.concatenate(predicted_labels,axis=0)

    def _predict_on_tensor(self,inputs):

        self.eval()

        def _argmax(tensor):
            return tensor.data.cpu().numpy().argmax(axis=1).reshape([-1,1])

        logits = self.__call__(inputs)
        labels=[_argmax(head) for head in logits]
        labels = np.hstack(labels)
        return labels,logits







