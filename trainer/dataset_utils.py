# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
import subprocess
import glob
import pickle

LOCAL_TRAIN_DIR='train'
LOCAL_TEST_DIR='test'

WORKING_DIR = os.getcwd()

def download_file_from_gcs(gcs_input_file,local_file_name):
  """Download files from GCS to a WORKING_DIR/.

  Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.

  Returns:
    A list to the local data paths where the data is downloaded.
  """

  # Copy raw files from GCS into local path.
  if gcs_input_file.startswith('gs'):
    subprocess.check_call(
          ['gsutil', 'cp', gcs_input_file, local_file_name])
    return local_file_name
  else:
    return gcs_input_file


def download_dir_from_gcs(gcs_input_dir,local_dir_name):
  """Download files from GCS to a WORKING_DIR/.

  Args:
    source: GCS path to the training data
    destination: GCS path to the validation data.

  Returns:
    A list to the local data paths where the data is downloaded.
  """

  # Copy raw files from GCS into local path.
  if gcs_input_dir.startswith('gs'):
    if not os.path.exists(local_dir_name):
      os.makedirs(local_dir_name)
    subprocess.check_call(
          ['gsutil', '-m', 'cp', '-r', os.path.join(gcs_input_dir,'*'), local_dir_name])
    return local_dir_name
  else:
    return gcs_input_dir



def _load_data(src_files_list, dst_dir):
  """Verifies if file is in Google Cloud.

  Args:
    path: (str) The GCS URL to download from (e.g. 'gs://bucket/file.csv')
    destination: (str) The filename to save as on local disk.

  Returns:
    A filename
  """

  if src_files_list[0].startswith('gs://'):
    if not os.path.exists(dst_dir):
      os.mkdir(dst_dir)
    destinations=[os.path.join(dst_dir,os.path.basename(s))for s in src_files_list]
    download_files_from_gcs(gcs_input_paths=src_files_list, local_file_names=destinations)
    return destinations
  return src_files_list

def _load_data_from_pkl(files_list):
  X=[]
  Y=[]
  for file in files_list:
    if os.path.exists(file):
      with open(file,'rb') as f:
        x,y,ids,classes=pickle.load(f)
      X.append(x)
      Y.append(y)

  X=np.concatenate(X,axis=0)
  X=np.expand_dims(X,axis=-1)
  Y=np.concatenate(Y,axis=0)
  return X,Y,classes


def prepare_data(train_files_dir, test_files_dir):
  """Loads MNIST Fashion files.

    License:
        The copyright for Fashion-MNIST is held by Zalando SE.
        Fashion-MNIST is licensed under the [MIT license](
        https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).

  Args:
    train_file: (str) Location where training data file is located.
    train_labels_file: (str) Location where training labels file is located.
    test_file: (str) Location where test data file is located.
    test_labels_file: (str) Location where test labels file is located.

  Returns:
    A tuple of training and test data.
  """
  train_files_list = _load_data(train_files_dir,LOCAL_TRAIN_DIR)
  test_files_list = _load_data(test_files_dir,LOCAL_TEST_DIR)

  x_train,y_train,classes=_load_data_from_pkl(train_files_list)
  x_test,y_test,classes=_load_data_from_pkl(test_files_list)

  return (x_train, y_train), (x_test, y_test),classes




