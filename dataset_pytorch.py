from torch.utils.data import Dataset

class BengaliDataset(Dataset):
    def __init__(self, images, image_ids, labels, transform_fn=None):
        self._images=images
        self._labels=labels
        self._image_ids=image_ids
        self._transform_fn=transform_fn

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        img,label,img_id=self._images[idx],self._labels[idx],self._image_ids[idx]
        if self._transform_fn:
            img=self._transform_fn(img)

        return {'image':img,'label':label,'image_id':img_id}



