import numpy as np
from torch.utils.data import Dataset
import cv2
import numpy as np
from consts import IMG_H,IMG_W

def _normalize_img(img):
    img=cv2.resize(img,(IMG_W,IMG_H))
    if len(img.shape)==2:
        img=np.expand_dims(img,axis=-1)
        img=np.concatenate([img,img,img],axis=2)
    else:
        if img.shape[-1]==1:
            img=np.concatenate([img,img,img],axis=2)

    eps=1e-3
    img=img.astype(np.float32)
    return (255-img)/255

class BengaliDataset(Dataset):
    def __init__(self, images, labels=None, transform_fn=None):
        self._images=images

        self._labels=labels
        self._transform_fn=transform_fn

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img=self._images[idx]
        label= -1 if self._labels is None else self._labels[idx]

        #img=np.concatenate([img,img,img],axis=-1)

        if self._transform_fn:
            img=self._transform_fn(img)

        img=_normalize_img(img)

        img_channel_first=np.transpose(img,[2,0,1])

        return {'image':img_channel_first,'label':label}



