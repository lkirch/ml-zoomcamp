#!/usr/bin/env python
# coding: utf-8

#import tensorflow.lite as tflite   # for testing locally
import tflite_runtime.interpreter as tflite  # for deploying
from keras_image_helper import create_preprocessor

preprocessor = create_preprocessor('xception', target_size=(224, 224))

interpreter = tflite.Interpreter(model_path='model2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

class_nms = [
    'normal',
    'fire',
    'smoke'
]

def predict(url):
    X = preprocessor.from_url(url)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    float_predictions = preds[0].tolist()

    return dict(zip(class_nms, float_predictions))

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result