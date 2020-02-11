import os
import numpy as np
import argparse
import subprocess
import tensorflow as tf
from .model_pytorch import Model
from .create_dataset_utils import load
from .score import calc_score
from .consts import MODELS_PRETRAINED_DIR, DATA_DIR,MODEL_NAME,BATCH_SIZE,EPOCHS, TRAIN_DATASET_DIR, VAL_DATASET_DIR, MODELS_DIR, METRIC_FILE_PATH,ARTIFACTS_DIR
from ..dataset_utils import download_dir_from_gcs, download_file_from_gcs
from .save_to_maybe_gs import save

debug_regime=False

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--ckpt_full_path',type=str,default='',help='if non empty model will be restored from ckpt state and continue training, path maye be in gs')
    parser.add_argument('--train_bin_files_dir',type=str,help=' train binary files in gs or local')
    parser.add_argument('--test_bin_files_dir',type=str,help=' test binary files in gs or local')
    parser.add_argument('--job-dir',type=str,default=ARTIFACTS_DIR,help=' directory for chekpoints and metric saving, is google storage directory for running in cloud')
    parser.add_argument('--verbosity',choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],default='INFO')
    args=parser.parse_args()
    return args


def main():

    args=parse_args()

    tf.logging.set_verbosity(args.verbosity)

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
    if (not model_dir.startswith('gs')) and (not os.path.exists(model_dir)):
        os.mkdir(model_dir)
    model_filepath=os.path.join(model_dir,MODEL_NAME)


    model=Model()

    model.compile(classes_list=classes)
    if not (args.ckpt_full_path==''):
        model_pretrained_filepath=download_file_from_gcs(args.ckpt_full_path,model_filepath)
        model.load(model_pretrained_filepath, classes)
    model.fit(train_images,train_labels, val_images,val_labels,path_to_model_save=model_filepath,batch_size=BATCH_SIZE,epochs=EPOCHS)

    model.eval()
    val_preds=model.predict( val_images)
    acc=[np.equal(val_pred,val_label) for val_pred,val_label in zip(val_preds,val_labels)]
    acc=np.array(acc,dtype=np.float32)
    acc=np.mean(acc,axis=0)
    tf.logging.info('validation accuracy = {}'.format(acc))

    score=calc_score(solution=val_preds,submission=val_labels)
    tf.logging.info('score={}'.format(score))

    def save_fn(path):
        with open(path,'wt') as file:
            file.write(str(score))

    save(save_fn,os.path.join(args.job_dir,METRIC_FILE_PATH))

if __name__ == "__main__":
    main()







