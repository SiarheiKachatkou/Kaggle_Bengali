import os
import numpy as np
import argparse
import subprocess
from .model_pytorch import Model
from .create_dataset_utils import load
from .score import calc_score
from .consts import MODELS_PRETRAINED_DIR, DATA_DIR,MODEL_NAME,BATCH_SIZE,EPOCHS, TRAIN_DATASET_DIR, VAL_DATASET_DIR, MODELS_DIR, METRIC_FILE_PATH,ARTIFACTS_DIR
from ..dataset_utils import download_dir_from_gcs
debug_regime=False

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_name',type=str,default='')
    parser.add_argument('--train_bin_files_dir',type=str,help=' train binary files in gs or local')
    parser.add_argument('--test_bin_files_dir',type=str,help=' test binary files in gs or local')
    parser.add_argument('--job-dir',type=str,default=ARTIFACTS_DIR,help=' directory for chekpoints and metric saving, is google storage directory for running in cloud')
    args=parser.parse_args()
    return args

def save(save_fn, dst_path_may_be_gs):
    tmp_name='tmp'
    save_fn(tmp_name)
    if dst_path_may_be_gs.startswith('gs:'):
        subprocess.check_call(
              ['gsutil',  'cp', tmp_name, dst_path_may_be_gs])
    else:
        os.rename(tmp_name, dst_path_may_be_gs)

def main():

    args=parse_args()

    local_train_dir=download_dir_from_gcs(args.train_bin_files_dir,DATA_DIR)
    local_test_dir=download_dir_from_gcs(args.test_bin_files_dir,DATA_DIR)

    train_images, train_labels, _, classes = load(local_train_dir)
    val_images, val_labels, _, _  = load(local_test_dir)

    if debug_regime:
        max_samples=100
        train_images=train_images[:max_samples]
        train_labels=train_labels[:max_samples]
        val_images=val_images[:max_samples]
        val_labels=val_labels[:max_samples]

    print('{} train images loaded'.format(len(train_images)))
    print('{} val images loaded'.format(len(val_images)))

    model_dir=os.path.join(args.job_dir,MODELS_DIR)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_filepath=os.path.join(model_dir,MODEL_NAME)


    model=Model()

    model.compile(classes_list=classes)
    if not (args.ckpt_name==''):
        model_pretrained_filepath=os.path.join(args.job_dir,args.ckpt_name)
        model.load(model_pretrained_filepath, classes)
    model.fit(train_images,train_labels, val_images,val_labels,path_to_model_save=model_filepath,batch_size=BATCH_SIZE,epochs=EPOCHS)

    save(model.save,model_filepath)

    model.eval()
    val_preds=model.predict( val_images)
    acc=[np.equal(val_pred,val_label) for val_pred,val_label in zip(val_preds,val_labels)]
    acc=np.array(acc,dtype=np.float32)
    acc=np.mean(acc,axis=0)
    print('validation accuracy = {}'.format(acc))

    score=calc_score(solution=val_preds,submission=val_labels)
    print('score={}'.format(score))

    def save_fn(path):
        with open(path,'wt') as file:
            file.write(str(score))

    save(save_fn,os.path.join(args.job_dir,METRIC_FILE_PATH))

if __name__ == "__main__":
    main()







