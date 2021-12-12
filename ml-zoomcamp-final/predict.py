#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import os

import tensorflow.lite as tflite
#import tflite_runtime.interpreter as tflite

from io import BytesIO
from urllib import request
from PIL import Image

# utilites to work with images
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess_input(x):
    x /= 127.5
    x -= 1.
    return x
    

# get the model
interpreter = tflite.Interpreter(model_path='model2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

#url = 'https://w2y9m3e9.stackpathcdn.com/wp-content/uploads/2020/06/house-on-fire-850x550.jpg'

def predict(url):
    
    # get the image
    img = download_image(url)
    img = prepare_image(img, target_size=(224, 224))
    # convert it to an array
    x = np.array(img, dtype='float32')
    X = np.array([x])
    # rescale the image from 256 colors
    X = preprocess_input(X)

    class_nms = ['normal','fire','smoke']
    
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()

    return dict(zip(class_nms, float_predictions))



def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)
    result = {
        'prediction': pred
    }

    return result 

result = predict(url)
print(result)
