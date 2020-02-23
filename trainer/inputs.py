import numpy as np
from functools import partial
import tensorflow as tf
from .consts import IMG_H,IMG_W,RECORD_BYTES,LABEL_BYTE_SIZE
tf.logging.set_verbosity(tf.logging.INFO)

def _zoom(x: tf.Tensor):
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Generate 20 crop settings, ranging from a 1% to 20% crop.
    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        # Create different crops for an image
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_ind=np.zeros(len(scales)), crop_size=(IMG_H, IMG_W))
        # Return a random crop
        return crops[tf.random_uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random_uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    # Only apply cropping 50% of the time
    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))

def augmentaze(x):
  x = tf.image.random_brightness(x, 0.05)
  x = tf.image.random_contrast(x, 0.7, 1.3)
  x=_zoom(x)
  return x

def parse_record_and_augm(raw_record,mode):#TODO add augmentation
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  img_size = IMG_W*IMG_H
  img = record_vector[0:img_size]
  img = tf.reshape(img, [IMG_H,IMG_W,1],name='parse_record_and_augm_reshape')

  img = tf.to_float(img)
  img=255-img
  img = (img)/ 128

  label_str=tf.substr(raw_record,img_size,LABEL_BYTE_SIZE)
  label=tf.decode_raw(label_str,tf.uint8)
  label = tf.cast(label, dtype=tf.int32)

  if mode==tf.estimator.ModeKeys.TRAIN:
    img=augmentaze(img)

  return (img, label)


def input_fn(filenames, batch_size, mode):

  """Input function.
  Args:
    features: (numpy.array) Training or eval data.
    labels: (numpy.array) Labels for training or eval data.
    batch_size: (int)
    mode: tf.estimator.ModeKeys mode

  Returns:
    A tf.estimator.
  """

  dataset=tf.data.FixedLengthRecordDataset(filenames,record_bytes=RECORD_BYTES)
  parse_record=partial(parse_record_and_augm,mode=mode)
  dataset=dataset.map(parse_record)
  if mode == tf.estimator.ModeKeys.TRAIN:
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)
  if mode in (tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT):
    dataset = dataset.batch(batch_size)


  return dataset


def serving_input_fn():
  """Defines the features to be passed to the model during inference.

  Expects already tokenized and padded representation of sentences

  Returns:
    A tf.estimator.export.ServingInputReceiver
  """
  feature_placeholder = tf.placeholder(tf.float32, [None, IMG_H, IMG_W,1])
  features = feature_placeholder
  return tf.estimator.export.TensorServingInputReceiver(features,
                                                        feature_placeholder)

if __name__=='__main__':
  import glob
  import os
  from get_args import get_args
  args=get_args()

  train_src_files_list=glob.glob(os.path.join(args.train_bin_files_dir,'*'))
  sess=tf.Session()
  ds=input_fn(train_src_files_list,10,tf.estimator.ModeKeys.TRAIN)
  iter=ds.make_one_shot_iterator()
  next_element=iter.get_next()
  while True:
    t_batch=sess.run(next_element)
    dbg=1
