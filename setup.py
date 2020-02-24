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
"""AI Platform package configuration."""
from setuptools import find_packages
from setuptools import setup
import os

REQUIRED_PACKAGES = ['requests==2.19.1',
                     'torch==1.2.0',
                     'torchvision==0.4.0',
                     'pytorch-lightning',
                     'pyarrow==0.15.0',
                     'sklearn',
                     'albumentations',
                     'pretrainedmodels',
                     'pandas',
                     'tqdm'
                     ]

setup(name='bengali',
      version='2.0',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,
      packages=find_packages(),
      description='Kaggle Bengali'
)

if True:
    os.system('git clone https://github.com/NVIDIA/apex; cd apex; pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./')

