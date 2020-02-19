import numpy as np
import os

p='/home/sergey/Downloads/handwriten_data/malayalam-handwritten-characters/datasetgray.npy'

begin_label=168

data=np.load(p,allow_pickle=True)
dbg=1
