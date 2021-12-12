#!/usr/bin/env python
# coding: utf-8

# Reading a saved model and testing on a sample image

import pandas as pd
import numpy as np
from numpy.random import seed
import os

from keras.models import load_model
from keras.preprocessing import image

base_dir = os.getcwd()
model_dir = base_dir + '/models'

# parameters
IMG_HEIGHT = IMG_WIDTH = 224

# load the saved model
model = load_model('models/model2-adam-convs')

imagePath = 'data/img_data/train/smoke/img_2036.jpg'

test_image = image.load_img(imagePath, target_size = (IMG_HEIGHT, IMG_WIDTH)) 
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

#predict the result
result = model.predict(test_image)

class_nms = ['normal','fire','smoke']

print(dict(zip(class_nms, result[0])))
