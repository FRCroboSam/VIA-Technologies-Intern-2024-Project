#same as the other version EXCEPT segment model is designed to work on square sized images 
#and trained on the output of the yolo bounding box model   

import math
import os

from PIL import Image
import matplotlib.pyplot as plt
import cv2
import csv
import keras as keras
import pandas as pd
import random
import shutil
import cv2
import matplotlib.image as mpimg
import ultralytics
from ultralytics import YOLO
import torch 
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Layer, Input
from moviepy.editor import ImageSequenceClip

#convert the segmentation model so you only need to divide by 255 once  into 

total_accuracy = 0
disease_only_accuracy = 0
detect_and_disease_accuracy = 0
total_leaves = 0

class_names = ['blight','citrus' ,'healthy', 'measles', 'mildew', 'mite', 'mold', 'rot', 'rust', 'scab', 'scorch', 'spot', 'virus']

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def no_background_image_tensor(mask, original_image):
    # Ensure mask and original image are TensorFlow tensors
    mask = tf.cast(mask, dtype=tf.float32)
    original_image = tf.cast(original_image, dtype=tf.float32)
    # Apply the mask to the original image
    mask = 1 - mask
    masked_image = tf.multiply(original_image, mask)
    return masked_image

def get_masked_image(mask, original_image):
    background_removed_image = no_background_image_tensor(create_mask(mask), original_image)
    return background_removed_image

@tf.keras.utils.register_keras_serializable()
class MaskBackgroundLayer(Layer):
    def compute_output_shape(self, input_shape):
        # The output shape is the same as the original image shape
        return input_shape[1]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def call(self, inputs):
        mask, input = inputs
        original_image = input
        masked_image = self.no_background_image_tensor(self.create_mask(mask), original_image)
        return masked_image

    def create_mask(self, pred_mask):
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]
    
    def no_background_image_tensor(self, mask, original_image):
        # Ensure mask and original image are TensorFlow tensors
        mask = tf.cast(mask, dtype=tf.float32)
        original_image = tf.cast(original_image, dtype=tf.float32)
        # Apply the mask to the original image
        mask = 1 - mask
        masked_image = tf.multiply(original_image, mask)
        return masked_image

    def get_masked_image(self, mask, original_image):
        background_removed_image = self.no_background_image_tensor(self.create_mask(mask), original_image)
        return background_removed_image
def compute_output_shape(self, input_shape):
    # The output shape is the same as the original image shape
    return input_shape[0]    

def build_mask_and_disease_model_pipeline(mask_background_model, disease_predict_model, IMG_SHAPE, base_learning_rate, mask_only):
    inputs = Input(shape=IMG_SHAPE)
    mask = mask_background_model(inputs / 255)
    x = inputs
    x = MaskBackgroundLayer()([mask, inputs])
    if(mask_only):
        mask_only_model = tf.keras.Model(inputs, x)
        mask_only_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        return mask_only_model
    else:
        x = disease_predict_model(x)  
        base_learning_rate = 0.001
        complete_model = tf.keras.Model(inputs,x)
        complete_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=['accuracy'])
        return complete_model 
    
def normalize_img(image):
    if(image.shape == (1, 128, 128, 3)):
        return image[0]
    else:
        return cv2.resize(image, (128, 128))

def show_image_with_prediction(image, prediction, confidence, correct):
    global subplot_index, fig
    if fig is None or subplot_index > 2:
        if fig is not None:
            plt.show()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        subplot_index = 1

    image = normalize_img(image)

    ax = fig.add_subplot(1, 2, subplot_index)
    ax.imshow(tf.keras.utils.array_to_img(image))
    ax.set_title(f"Prediction: {prediction} ({confidence:.2f}), Correct: {correct}")
    ax.axis('off')
    ax.set_xticks([])  # Remove x-axis ticks
    ax.set_yticks([])  # Remove y-axis ticks
    ax.set_xticklabels([])  # Remove x-axis tick labels
    ax.set_yticklabels([])  # Remove y-axis tick labels
    for spine in ax.spines.values():  # Turn off the spines (border lines)
        spine.set_visible(False)
    subplot_index += 1


def leaf_detect(img, model):
    detect_result = model(img, conf=0.5)
    detect_img = detect_result[0].plot()
    # detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)    
    return detect_img, detect_result

def find_best_bounding_box(detect_result):
    largest_area = 0
    best_bbox = None
    bestArea = 0 
    for r in detect_result:
        for box in r.boxes:
            bbox = box.xywh.tolist()[0]
            # Save the x, y, width, and height to separate variables and round them to the nearest whole numbers
            x, y, w, h = map(round, bbox)  
            area = w * h
            if(area > bestArea):
                best_bbox = box 
                bestArea = area 
    if(bestArea == 0):
        return None
    else:
        return best_bbox.xywh.tolist()[0]
def resize_bbox_img_to_square(bbox, bbox_image):
    original_bbox = bbox
    h = bbox_image.shape[0]
    w = bbox_image.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if (w > h):
        pad_top = int((w - h) // 2)
        pad_bottom = w - h - pad_top    

        bbox_image = cv2.copyMakeBorder(bbox_image, pad_top, pad_bottom, 0, 0, 0, value=[0, 0, 0])

    elif (h > w):
        pad_right = int((h - w) // 2)
        pad_left = h - w - pad_right

        bbox_image = cv2.copyMakeBorder(bbox_image, 0, 0, pad_left, pad_right, 0, value=[0, 0, 0])
    return bbox_image
    
def get_all_bounding_boxes(image, model):
    # print("BEFORE")
    # print(image.shape)
    # #image = cv2.resize(image, (640, 480))
    # print(image.shape)

    detect_img, detect_result = leaf_detect(image, model)

    bbox_images = []
    bboxes = []

    copy = image
    # print("STARTING THE DETECT")
    # print(copy.shape)
    for r in detect_result:
        for box in r.boxes:
            bbox = box.xywh.tolist()[0]
            # Save the x, y, width, and height to separate variables and round them to the nearest whole numbers
            x, y, w, h = map(round, bbox)  
            copy = np.zeros_like(image)
            min_y =  int(round(y - 0.5 * h))
            max_y = int(round(y + 0.5 * h))
            min_x = int(round(x - 0.5 * w))
            max_x = int(round(x + 0.5 * w))
            # w = min(w, 640 - min_x)
            # h = min(h, 480 - min_y)

            bbox_image =  image[min_y:max_y, min_x:max_x]
            bbox_image = resize_bbox_img_to_square(bbox, bbox_image)
            bbox_image = cv2.resize(bbox_image, (128, 128))
            new_bbox = [min_x, min_y, w, h]
            # printBbox(new_bbox)
            bboxes.append([min_x, min_y, w, h])
            bbox_images.append(bbox_image)
    return detect_img, bbox_images, bboxes

def return_prediction(disease_predict_model, image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    prediction = disease_predict_model.predict(np.expand_dims(image, axis=0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    prediction_array = np.array(prediction)


    max_indices = np.argsort(prediction_array[0])[-2:]  # Get indices of top two predictions
    max_indices = max_indices[::-1]  # Reverse to get highest to lowest

    max_index = max_indices[0]
    predicted_class = class_names[max_index]

    confidence = prediction[0, max_index]
    # print("Predicted_class: " + predicted_class)
    # print("Confidence: " + str(float(confidence)))
    return predicted_class, confidence

def return_predictions(bbox_images, disease_predict_model):
    predictions = []
    confidences = []
    for bbox_image in bbox_images:
        predicted_class, confidence = return_prediction(disease_predict_model, bbox_image)
        if(confidence < 0.90):
            print("OVERRIDE TO BE HEALTHY")
            predicted_class = 'healthy'
        predictions.append(predicted_class)
        confidences.append(confidence)
    return predictions, confidences

def label_image_with_multiple_bbox(image, disease_predict_model, bbox_images, bboxes, predictions=[], confidences=[]):
    labeled_image = image.copy()
    # predictions = []
    # confidences = []
    # print("NUM PREDICTIONS ACTUAL: " + str(len(predictions)))
    # print("NUM BBOXES: " + str(len(bbox_images)))
    total_leaves = len(bbox_images)
    healthy = 0
    if(len(predictions) == 0):
        for bbox_image in bbox_images:
            predicted_class, confidence = return_prediction(disease_predict_model, bbox_image)
            predictions.append(predicted_class)
            confidences.append(confidence)
    index = 0
    for bbox_image in bbox_images:
        x, y, width, height = bboxes[index]
        prediction = predictions[index]
        confidence = confidences[index]
        color = (255, 0, 0)
        if(prediction == "healthy"):
            healthy += 1
            color = (0, 255, 0)
        labeled_image = cv2.rectangle(labeled_image, (x, y), (x + width, y + height), color, 1)

        # Prepare label text
        label_text = f'{prediction} ({confidence:.2f})'
        font = cv2.FONT_HERSHEY_TRIPLEX
        font_scale = 0.7
        thickness = 1

        # # Get the text size
        (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

        # Draw the white rectangle
        cv2.rectangle(labeled_image, (x+3, y), (x + text_width + 5, y + baseline + 10), (255, 255, 255), cv2.FILLED)

        # Put text
        cv2.putText(labeled_image, label_text, (x + 5, y + 15), cv2.FONT_HERSHEY_TRIPLEX,
                                    0.8, color, 1, 1)
        index += 1
    
    #add a label at the top left of the image 
    status = 'DISEASED'
    color = (255, 0, 0)
    percentage = int((float(healthy) / total_leaves) * 100)
    status = f" Status: DISEASED ({percentage}%)"

    if(healthy > 0.8 * total_leaves):
        status = f" Status: HEALTHY ({percentage}%)"
        color = (0, 255, 0)
    elif (healthy >= 0.5 * total_leaves):
        status = f"Village Status: NEUTRAL ({percentage}%)"
        color = (204, 204, 0) #yellow
    elif (healthy >= 0.3 * total_leaves):
        status = f"Village Status: WARNING ({percentage}%)"
        color = (255, 165, 0) #orange 

    # Draw the white rectangle
    cv2.rectangle(labeled_image, (x+3, y), (x + text_width * 2 + 5, y + baseline + 20), (255, 255, 255), cv2.FILLED)
        
    cv2.putText(labeled_image, status, (x + 10, y + 10), cv2.FONT_HERSHEY_TRIPLEX,
                                    1.5, color, 2, 1)
    return labeled_image





def label_image(image, prediction, confidence, bbox):
    labeled_image = image.copy()
    x, y, width, height = bbox
    color = (255, 0, 0)
    if(prediction == "healthy"):
        color = (0, 255, 0)

    # Draw rectangle
    labeled_image = cv2.rectangle(labeled_image, (x, y), (x + width, y + height), color, 2)

    # Prepare label text
    label_text = f'{prediction} ({confidence:.2f})'
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 0.4
    thickness = 1

    # Get the text size
    (text_width, text_height), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

    # Draw the white rectangle
    cv2.rectangle(labeled_image, (x+3, y), (x + text_width + 8, y + baseline + 10), (255, 255, 255), cv2.FILLED)

    # Put text
    cv2.putText(labeled_image, label_text, (x + 5, y + 10), cv2.FONT_HERSHEY_TRIPLEX,
                                 0.4, color, 1, 1)

    return labeled_image

def find_last_root_contour(hierarchy):
    last_root_index = -1
    for i in range(len(hierarchy[0])):
        if hierarchy[0][i][3] == -1:
            last_root_index = i
    
    return last_root_index
def printBbox(bbox):
    print("PRINTING BBOX FOIJKL")
    print("X: " + str(bbox[0]))
    print("Y: " + str(bbox[1]))
    print("WIDTH: " + str(bbox[2]))
    print("HEIGHT: " + str(bbox[3]))
def resize_mask_to_bbox(bbox, bbox_img, mask):

    x, y, w, h = bbox
    # printBbox(bbox)
    resized_mask = np.repeat(mask, 3, axis=2)  # Shape: (128, 128, 3)

    resized_mask = np.zeros((resized_mask.shape[0], resized_mask.shape[1], 3)) + np.array(resized_mask)
    final_mask = np.ones((IMG_SHAPE[0], IMG_SHAPE[1], 1))
    # print("FINAL MASK SHAPE")
    # print(final_mask.shape)
    actual_final_shape = final_mask[y:y+h, x:x+w].shape

    w = actual_final_shape[1]
    h = actual_final_shape[0]
    # print("ACTUAL FINAL SHAPE")
    # print(actual_final_shape)

    if (w > h):
        #resize it to be the version with padding added to make it a square
        resized_mask = cv2.resize(resized_mask, (w, w))

        resized_mask = resized_mask[:, :, 0:1]
     

        pad_top = int((w - h) // 2)
        pad_bottom = w - h - pad_top
        if(pad_bottom == 0):
            resized_mask = resized_mask[pad_top:, :]
        elif(pad_top == 0):
            resized_mask = resized_mask[:-pad_bottom, :]
        else:
            resized_mask = resized_mask[pad_top:-pad_bottom, :]
    elif (h > w):
        resized_mask = cv2.resize(resized_mask, (h, h))
        resized_mask = resized_mask[:, :, 0:1]


        pad_right = int((h - w) // 2)
        pad_left = h - w - pad_right

        if(pad_right == 0):
            resized_mask = resized_mask[:, pad_left:]
        elif(pad_left == 0):
            resized_mask = resized_mask[:, :-pad_right]
        else:
            resized_mask = resized_mask[:, pad_left:-pad_right]
    else:
        resized_mask = cv2.resize(resized_mask, (h, w))
        resized_mask = resized_mask[:, :, 0:1]

    final_mask[y:y+h, x:x+w] = resized_mask
    original_size_mask = resized_mask
    # print("RESIZED MASK SHAPE")
    # print(resized_mask.shape)
    return final_mask, original_size_mask

    # bboxes.append([min_x, min_y, max_x, max_y])
def return_prediction_for_image(image, bbox_model, mask_only_model, mask_background_model, mask_disease_model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    bbox = [0, 0, 128, 128]
    prediction = ''
    confidence = 0
    if(image.shape != (128, 128, 3) or image.shape != (1, 128, 128, 3)):
        #image = normalize_img(image) 
        if bbox_model is not None:
            detect_img, bbox_images, bboxes = get_all_bounding_boxes(image.copy(), bbox_model)

        
            if(mask_background_model is not None):
                masks = []
                resized_masks = []
                for bbox, bbox_img in zip(bboxes, bbox_images):
                    mask = mask_only_model.predict(bbox_img[tf.newaxis, ...] / 255)    
                    mask = create_mask(mask)
                    mask, original_mask = resize_mask_to_bbox(bbox, bbox_img, mask)
                    masks.append(mask)
                    resized_masks.append(original_mask)
                # print("NUM BBOX IMAGES ACTUAL: " + str(len(bbox_images)))
                predictions, confidences = return_predictions(bbox_images, mask_disease_model)
                # print("NUM PREDICTIONS ACTUAL BEFORE MASK: " + str(len(predictions)))

                image = overlay_multiple_masks(resized_masks, bboxes, bbox_images, image, predictions)
                # print("NUM PREDICTIONS ACTUAL AFTER MASK: " + str(len(predictions)))

                # image = overlay_mask_on_image(mask, image, color)
                image = label_image_with_multiple_bbox(image, 
                                                        mask_disease_model, bbox_images, bboxes,
                                                        predictions, confidences)
                

            else:
                # image = detect_img
                predictions = []
                confidences = []
                image = label_image_with_multiple_bbox(image, 
                                        mask_disease_model, bbox_images, bboxes,
                                        predictions, confidences)
        else:
            prediction, confidence = return_prediction(mask_disease_model, image)
            # show_image_with_prediction(image, prediction, confidence, correct)
        if not((bbox_model is not None)):
            image = label_image(image, prediction, confidence, bbox)

        return image, prediction, confidence
    return "None", 0.0, 0


def show_images(names, images, final_frames, predictions, confidences, correct):
    num_images = len(images)
    plt.figure(figsize=(15, 5))  # Adjust the figure size as needed
    name = names[0] + ".jpg"
    image = images[0]
    print(image.shape)
    final_frames.append(image)
    final_frames.append(image)

    image = tf.keras.utils.array_to_img(image)

    file_path = os.path.join('leaf_detect_results', name)

    image.save(file_path)

    plt.tight_layout()
    plt.show()
def overlay_multiple_masks(masks, bboxes, bbox_imgs, original_image, predictions, color=[0, 255, 0], alpha=0.4):
    for mask, bbox, bbox_img, prediction in zip(masks, bboxes, bbox_imgs, predictions):
        x, y, w, h = bbox 
        #printBbox(bbox)
        if(prediction == "healthy"):
            color = [0, 255, 0]
        else:
            color = [255, 0, 0]
        bbox_img = original_image[y:y+h, x: x+w]
        # print("BBOX IMG SHAPE")
        # print(bbox_img.shape)
        bbox_img = overlay_mask_on_image(mask, bbox_img, color)
        original_image[y:y+h, x: x+w] = bbox_img


    return original_image 

def overlay_mask_on_image(mask, original_image, color=[0, 255, 0], alpha=0.2):
    # Ensure mask and original image are TensorFlow tensors
    green_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    green_mask[:, :, 0] = color[0] 
    green_mask[:, :, 1] = color[1]
    green_mask[:, :, 2] = color[2]

    original_image = np.array(original_image)    
    original_image = original_image #/ 255 / 255
    non_zero_mask = np.broadcast_to(mask != 1, original_image.shape)
    other_mask = np.broadcast_to(mask == 1, original_image.shape)
    # green_mask[non_zero_mask] = 255
    green_mask[other_mask] = 255

    original_image = cv2.addWeighted(original_image, 1 - alpha, green_mask, alpha, 0)
    return original_image; 
    
def save_image(name, plt):

    file_path = os.path.join('leaf_detect_results', name)
    plt.savefig(file_path)

def perform_detection(video_frame, frame_num, final_frames, leaf_detect_model, mask_only_model, 
                  mask_background_model, mask_and_predict_model, disease_only_model):   
    image = video_frame
    image = cv2.resize(image, (IMG_SHAPE[1], IMG_SHAPE[0]))

    if image is None:
        print(f"Error: Unable to read image from video frame")
    else:
        names = [str(frame_num)]
        predictions = []
        confidences = []
        images = []
        image1, prediction1, confidence1 = return_prediction_for_image(image, leaf_detect_model, mask_only_model,
                                                                                    mask_background_model, mask_and_predict_model, 
                                                                                    )
        images.append(image1)
        predictions.append(prediction1)
        show_images(names, images, final_frames, predictions, confidences, 0)

def evaluate_accuracy():
    print("TOTAL CORRECT: " + str(total_accuracy) + " Out of: " + str(total_leaves))
    print("DISEASE ONLY CORRECT: " + str(disease_only_accuracy) + " Out of: " + str(total_leaves))
    print("Detect And Disease Accuracy " + str(detect_and_disease_accuracy ) + " Out of: " + str(total_leaves))

def main(video_frame, frame_num, final_frames):
    IMG_SHAPE = (video_frame[1], video_frame[0], 3)
    base_learning_rate = 0.001

    perform_detection(video_frame, frame_num, final_frames, leaf_detect_model, masking_model, mask_background_model,
                      mask_and_predict_model, disease_predict_model)
    


leaf_detect_model_path = 'runs/detect/train59/weights/best.pt'

masking_model_path = 'for_yolo_v2.keras'

disease_predict_model_path = 'trained_models/disease_detect_models/detection.keras'
disease_predict_model_path = 'det_seg_trans_aug_v2.keras'
leaf_detect_model = YOLO(leaf_detect_model_path)
masking_model = tf.keras.models.load_model(masking_model_path)


disease_predict_model = tf.keras.models.load_model(disease_predict_model_path)

IMG_SHAPE = (128, 128, 3)
base_learning_rate = 0.001

mask_background_model = build_mask_and_disease_model_pipeline(masking_model, disease_predict_model, IMG_SHAPE, base_learning_rate,True)
mask_and_predict_model = build_mask_and_disease_model_pipeline(masking_model, disease_predict_model, IMG_SHAPE, base_learning_rate, False)

cap = cv2.VideoCapture("IMG_9955.MOV")
print("GOT VIDEO CAPTURE")

frame_num = 0

if(os.path.exists('leaf_detect_results')):
    shutil.rmtree('leaf_detect_results')
os.makedirs('leaf_detect_results')
IMG_SHAPE = (480, 850, 3)

final_frames = []
video_input_folder = "../Samuel_Plant_Disease/datasets/PlantVillage/video"
if not os.path.exists(video_input_folder):
    os.makedirs(video_input_folder)
while True and frame_num < 60*30:
    result, video_frame = cap.read()
    if(frame_num % 2 == 0):
        # video_frame = cv2.resize(video_frame, (640, 480))
        # print(video_frame.shape)
        # image = tf.keras.utils.array_to_img(video_frame)
        # file_path = os.path.join(video_input_folder, 'frame' + str(frame_num) + '.jpg')
        # image.save(file_path)
        main(video_frame, 'frame' + str(frame_num), final_frames)
        print("FRAME IS: " + str(frame_num))
    frame_num += 1
    # cv2.imshow("VIA Disease Detect Video Demo", video_frame)
    # cv2.waitKey()
    # if(cv2.waitKey(1) & 0xFF == ord("q")):
    #     break


cap.release()

print("DONE GETTING THE FRAMES")
# print(len(final_frames))
clip = ImageSequenceClip(final_frames, fps=30)

# Write the result to a video file
clip.write_videofile("output_video.mp4", codec="libx264")


'''
TODO: See if all the bbox pictures result in the same prediction from leaf_detect_segment.py"
try running this one 
'''
