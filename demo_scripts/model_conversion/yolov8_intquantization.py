# import math
# import os

# from PIL import Image
# import matplotlib.pyplot as plt
# import cv2
# import csv
# #import keras as keras
# import pandas as pd
# import random
# import shutil
# import cv2
# import matplotlib.image as mpimg
# import ultralytics
# from ultralytics import YOLO
# import torch 
# from PIL import Image
# import numpy as np
# from tensorflow.keras.preprocessing.image import array_to_img, load_img, img_to_array
# import keras

# import tensorflow as tf
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer, Input
# import onnx_tf
# import onnx

import tensorflow as tf
import os
import numpy as np 
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pathlib


img_height = 128
img_width = 128

def load_image(image_path, mask_path):
    image_path = image_path.numpy().decode('utf-8')
    mask_path = mask_path.numpy().decode('utf-8')
    
    image = load_img(image_path, target_size=(img_height, img_width))
    image = img_to_array(image) / 255.0
    
    mask = load_img(mask_path, target_size=(img_height, img_width), color_mode="grayscale")
    mask = img_to_array(mask) / 255.0
    return image, mask

def load_dataset(image_folder, mask_folder):
    image_paths = sorted([os.path.join(image_folder, fname) for fname in os.listdir(image_folder)])
    mask_paths = sorted([os.path.join(mask_folder, fname) for fname in os.listdir(mask_folder)])
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
    def process_path(image_path, mask_path):
        image, mask = tf.py_function(load_image, [image_path, mask_path], [tf.float32, tf.float32])
        image.set_shape([img_height, img_width, 3])
        mask.set_shape([img_height, img_width, 1])
        return image, mask
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).take(100):
    input_value = np.expand_dims(input_value, axis=0)
    print(input_value.shape)

    yield [input_value]

def convert_model(model):
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model_quant = converter._convert_from_saved_model()
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    directory_name = 'quant_models'

    # Create the directory if it does not exist
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f"Directory '{directory_name}' created successfully.")
    else:
        print(f"Directory '{directory_name}' already exists.")

    tflite_model_quant_file = os.path.join(directory_name,"mask_and_predict.tflite")
    with open(tflite_model_quant_file, 'wb') as f:
        f.write(tflite_model_quant)  # Assuming tflite_quant_model contains the quantized model bytes

plant_village = "datasets/PlantVillage"
unlabeled_dir = os.path.join(plant_village, 'unlabeled')
masked_dir = os.path.join(plant_village, 'segmented')


# mask_and_predict_model = tf.keras.models.load_model("for_yolo_v2.keras")
# mask_and_predict_model.save("actual_saved_model")
save_model = "actual_saved_model"
mask_train_dataset = load_dataset(unlabeled_dir, masked_dir)
mask_train_batches = mask_train_dataset.batch(10)

train_images = []
 
for image, mask in mask_train_batches.unbatch().take(4):
    print(image.shape)
    train_images.append(image.numpy())
 
train_images = tf.convert_to_tensor(train_images)
convert_model(save_model)
