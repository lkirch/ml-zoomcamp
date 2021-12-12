#!/usr/bin/env python
# coding: utf-8

# Note: If you need to run this more than once, you will need to either 
#       move all the validation files back to training or 
#       just start fresh with the data from the original source data found int original-data.

# data_url = "https://www.kaggle.com/ritupande/fire-detection-from-cctv/download/archive.zip"

import pandas as pd
import numpy as np
from numpy.random import seed
import os
import shutil
import random
import zipfile

seed(42)
np.random.seed(42)

os.system('mkdir -p ./data/img_data/val')
os.system('mkdir -p ./data/img_data/val/default')
os.system('mkdir -p ./data/img_data/val/fire')
os.system('mkdir -p ./data/img_data/val/smoke')

base_dir = os.getcwd()

train_dir = base_dir + '/data/img_data/train'
train_dir_default = base_dir + '/data/img_data/train/default'
train_dir_fire = base_dir + '/data/img_data/train/fire'
train_dir_smoke = base_dir + '/data/img_data/train/smoke'

val_dir = base_dir + '/data/img_data/val'
val_dir_default = base_dir + '/data/img_data/val/default'
val_dir_fire = base_dir + '/data/img_data/val/fire'
val_dir_smoke = base_dir + '/data/img_data/val/smoke'

test_dir = base_dir + '/data/img_data/test'
test_dir_default = base_dir + '/data/img_data/test/default'
test_dir_fire = base_dir + '/data/img_data/test/fire'
test_dir_smoke = base_dir + '/data/img_data/test/smoke'

# randomly select 20% of the files and move to the validation directories
# https://stackoverflow.com/questions/49280966/pulling-random-files-out-of-a-folder-for-sampling

default_fnames = os.listdir(train_dir_default)
fire_fnames = os.listdir(train_dir_fire)
smoke_fnames = os.listdir(train_dir_smoke)

default_random_files = np.random.choice(default_fnames, int(len(default_fnames)*.2), replace=False)
fire_random_files = np.random.choice(fire_fnames, int(len(fire_fnames)*.2), replace=False)
smoke_random_files = np.random.choice(smoke_fnames, int(len(smoke_fnames)*.2), replace=False)

# move the validation files
for filename in default_random_files:
    shutil.move(os.path.join(train_dir_default, filename), os.path.join(val_dir_default, filename))
    
for filename in fire_random_files:
    shutil.move(os.path.join(train_dir_fire, filename), os.path.join(val_dir_fire, filename))
    
for filename in smoke_random_files:
    shutil.move(os.path.join(train_dir_smoke, filename), os.path.join(val_dir_smoke, filename))
    
    
print('Number of default training images:', len(os.listdir(train_dir_default)))
print('Number of fire training images:', len(os.listdir(train_dir_fire)))
print('Number of smoke training images:', len(os.listdir(train_dir_smoke)))
print()
print('Number of default validation images:', len(os.listdir(val_dir_default)))
print('Number of fire validation images:', len(os.listdir(val_dir_fire)))
print('Number of smoke validation images:', len(os.listdir(val_dir_smoke)))
print()
print('Number of default test images:', len(os.listdir(test_dir_default)))
print('Number of fire test images:', len(os.listdir(test_dir_fire)))
print('Number of smoke test images:', len(os.listdir(test_dir_smoke)))