import numpy as np
from torch.utils.data import Dataset

def _normalize_img(img):
    eps=1e-3
    img=img.astype(np.float32)
    return (img-np.mean(img))/(np.std(img)+eps)

class BengaliDataset(Dataset):
    def __init__(self, images, labels, transform_fn=None):
        self._images=[_normalize_img(img) for img in images]
        self._labels=labels
        self._transform_fn=transform_fn

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img,label=self._images[idx],self._labels[idx]
        img=np.concatenate([img,img,img],axis=0)
        if self._transform_fn:
            img=self._transform_fn(img)

        return {'image':img,'label':label}



