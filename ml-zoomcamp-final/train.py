#!/usr/bin/env python
# coding: utf-8

# Reproducibility Note: Due to randomness you will may get different results.

# Based on analysis in our jupyter notebook, the script will create and save a model using 
# the Adam optimizer with a learning rate of 0.001 and additonal Conv2D layers 


import pandas as pd
import numpy as np
from numpy.random import seed
import os
import shutil
import random

from skimage.io import imread, imshow

#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import random
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
#import tensorflow_hub as hub

import tensorflow.lite as tflite

from PIL import Image

import warnings
warnings.filterwarnings('ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# pre-processing

seed(42)
np.random.seed(42)
tf.random.set_seed(42)

base_dir = os.getcwd()

train_dir = base_dir + '/data/img_data/train'
train_dir_default = train_dir + '/default'
train_dir_fire = train_dir + '/fire'
train_dir_smoke = train_dir + '/smoke'

val_dir = base_dir + '/data/img_data/val'
val_dir_default = val_dir + '/default'
val_dir_fire = val_dir + '/fire'
val_dir_smoke = val_dir + '/smoke'

test_dir = base_dir + '/data/img_data/test'
test_dir_default = test_dir + '/default'
test_dir_fire = test_dir + '/fire'
test_dir_smoke = test_dir + '/smoke'

# parameters for the ImageDataGenerator
BATCH_SIZE = 10
IMG_WIDTH = 224
IMG_HEIGHT = 224

# model parameters
BATCH_SIZE = 10    # how many images to process in each batch
IMG_CHANNELS = 3   # how many channels (Red Green Blue = RGB)
EPOCHS = 10        # how many times we want to pass over the all the training data
STEP_SIZE = 10     # number of steps


# pre-processing
train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    target_size=(IMG_HEIGHT, IMG_WIDTH)
)


val_datagen = ImageDataGenerator(rescale = 1./255)

validation_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    batch_size = BATCH_SIZE,
    class_mode = 'categorical',
    target_size = (IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)

test_datagen = ImageDataGenerator(rescale = 1./255)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=False
)


# functions to create models
def create_model(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS, optimizer='Adam', lr=0.001, momentum=0.8):
    """ function to create a basic neural network model given image height, image width, number of image channels,
        optimizer, learning rate and momentum
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())                       # this converts it a vector 1D
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    if optimizer == 'SGD':
        optimizer=optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        optimizer=optimizers.Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
    return model


def create_drop_model(learning_rate=0.001, optimizer='SGD', droprate=0.0, img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS):
    """ function to create a basic neural network model given image height, image width, number of image channels,
        optimizer, learning rate, momentum, and dropout rate
    """
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())                       
    model.add(layers.Dense(6, activation='relu'))
    model.add(layers.Dropout(droprate))   
    model.add(layers.Dense(3, activation='softmax'))
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    return model


def create_model_with_more_conv_layers(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS, 
                                       optimizer='Adam', lr=0.001, momentum=0.8):
    """ function to create a basic neural network model that has additional Convolutional 2D layers"""
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3, activation='softmax'))
    
    if optimizer == 'SGD':
        optimizer=optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        optimizer=optimizers.Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
    return model


def create_model_with_more_dense_layers(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS, 
                                        optimizer='Adam', lr=0.001, momentum=0.8):
    """ function to create a basic neural network model that has additional Convolutional 2D and Dense layers"""

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(3, activation='softmax'))
    
    if optimizer == 'SGD':
        optimizer=optimizers.SGD(learning_rate=lr, momentum=momentum)
    else:
        optimizer=optimizers.Adam(learning_rate=lr)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
    return model


def train_nn(img_height=IMG_HEIGHT, img_width=IMG_WIDTH, img_channels=IMG_CHANNELS):
    # build model
    model = create_model_with_more_conv_layers(img_height, img_width, img_channels, optimizer='Adam', lr=0.001, momentum=0.0)
    # train model
    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=validation_generator,
                        validation_steps=STEP_SIZE)
    print('Adam optimizer with learning_rate = 0.001')
    print('Median Training Accuracy: ', np.median(history.history['acc']))
    print('Standard Deviation Training Loss: ', np.std(history.history['loss']))
    print('Median Validation Accuracy: ', np.median(history.history['val_acc']))
    print('Standard Deviation Validation Loss: ', np.std(history.history['val_loss']))
    return model


model = train_nn(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)
model.save('models/model2-adam-convs-from-script')   # just saving the model for later if we need to run locally

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open('model2.tflite', 'wb') as f_out:
  f_out.write(tflite_model)
