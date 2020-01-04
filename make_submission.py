import pandas as pd
import os
import numpy as np
from model import Model
from image_data_generator import ImageDataGenerator
from consts import TARGETS, DATA_DIR,MODEL_NAME, TEST_CSV, RAW_DIR, TEST_DATASET_PKL, IMAGE_GEN_PKL, MODELS_DIR, SUBMISSION_DIR, SUBMISSION_CSV
from create_dataset import load

if __name__ == "__main__":

    test_ids=pd.read_csv(os.path.join(DATA_DIR,RAW_DIR,TEST_CSV))

    imgs, labels, ids, classes = load(os.path.join(DATA_DIR,TEST_DATASET_PKL))
    gen=ImageDataGenerator.load(os.path.join(DATA_DIR,IMAGE_GEN_PKL))

    model=Model()
    model.load(os.path.join(DATA_DIR,MODELS_DIR,MODEL_NAME),classes)
    preds = model.predict(gen,imgs)

    row_ids=[]
    targets=[]
    for pred,id in zip(preds,ids):
        for target_idx,t in enumerate(TARGETS):
            row_id=id+'_'+t
            row_ids.append(row_id)
            targets.append(pred[target_idx])


    idxs=[row_ids.index(i) for i in test_ids['row_id']]

    row_ids=np.array(row_ids)
    targets=np.array(targets)
    targets=targets[idxs]
    row_ids=row_ids[idxs]
    submission=pd.DataFrame(data={'row_id':row_ids,'target':targets})
    submission.to_csv(os.path.join(DATA_DIR,SUBMISSION_DIR,SUBMISSION_CSV),index=False)




