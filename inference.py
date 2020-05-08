#!/usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import colorsys
import os, sys, argparse
import cv2
import time, json
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Lambda
from tensorflow_model_optimization.sparsity import keras as sparsity
from PIL import Image

from yolo3.model import get_yolo3_model, get_yolo3_inference_model, get_yolo3_prenms_model
from yolo3.postprocess_np import yolo3_postprocess_np
from yolo3.data import preprocess_image, letterbox_image
from yolo3.utils import get_classes, get_anchors, get_colors, draw_boxes, touchdir
from confusion_matrix import compute_confison_matrix, alter_predicted_csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tensorflow.keras.utils import multi_gpu_model

# Setting up Model Configuration
model_type = 'darknet'
model_path = 'inference_model/shelf_product_detection_inference_model.h5'
anchors_path = 'configs/tiny_yolo_anchors.txt'
classes_path = 'yolo_format_training_classes.txt'
model_image_size = (416, 416)
gpu_num = 1

# Load Classes and get anchors for comparision
class_names = get_classes(classes_path)
anchors = get_anchors(anchors_path)
colors = get_colors(class_names)

num_anchors = len(anchors)
num_classes = len(class_names)
num_feature_layers = num_anchors//3

# Initialise the model with same architecture and load the trained model provided in the config
yolo_model, _ = get_yolo3_model(model_type, num_feature_layers, num_anchors, num_classes, input_shape=model_image_size + (3,))
yolo_model.load_weights(model_path) # make sure model, anchors and classes match
print('{} model, anchors, and classes loaded.'.format(model_path))

# Define the Prediction with config parameters
def predict(image_data, image_shape):
    
    out_boxes, out_classes, out_scores = yolo3_postprocess_np(yolo_model.predict(image_data), 
                                                              image_shape, 
                                                              anchors,
                                                              len(class_names), 
                                                              model_image_size, 
                                                              max_boxes=100, confidence = 0.3, iou_threshold=0.4)
    
    return out_boxes, out_classes, out_scores


# Initialise the inference output path
txt_save_path = './results/txt/'
images_save_path = './results/images/'

if not os.path.exists(txt_save_path):
    os.makedirs(txt_save_path)

if not os.path.exists(images_save_path):
    os.makedirs(images_save_path)

# Load the images for inference
images_path = './test/test_images/'
test_images = os.listdir(images_path)

# Similarly load the pre-computed groundtruth for the same test images to benchmark the prediction
ground_truth_df = pd.read_csv('./test/test_groundtruth.csv')

# Initialising variables to save the output values
Images2Product = {} # To map the number of products detected for all given set of images
Images2BoundingBox = [] # To save all images predicted bounding box
  

for image in sorted(test_images):
    
    cur_image_prediction = [image] #To save each image's predicted boxes
    
    print("Processing Image : {}".format(image))

    image_path = images_path+image

    cv2_image  = cv2.imread(image_path)
    pil_image = Image.open(image_path)

    image_data = preprocess_image(pil_image, model_image_size)
    image_shape = pil_image.size

    start = time.time()

    out_boxes, out_classes, out_scores = predict(image_data, image_shape)

    end = time.time()

    Images2Product[image] = len(out_boxes)

    file = open(txt_save_path+image.split(".")[0]+".txt", 'w')

    for index, box in enumerate(out_boxes):
        
        conf = round(out_scores[index]*100)

        cv2.rectangle(cv2_image,(int(box[0]), int(box[1])), (int(box[2]), int(box[3])) , (0,255,255), 2 )

        cv2.putText(cv2_image, str("Conf:{}".format(conf)), (int(box[0]-5), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,255,255), 2, cv2.LINE_AA)

        cur_image_prediction.extend(box) # Adding the current predicted box to cur image's prediction

        box = [str(x) for x in box]
        
        file.write(",".join(box)+"\n")

    file.close()

    cv2.imwrite(images_save_path+image,cv2_image)
    
    Images2BoundingBox.append(cur_image_prediction)


print("\nPredicted images are saved in './results/images/' folder") 

predicted_bbox_df = pd.DataFrame(data = Images2BoundingBox)

predicted_bbox_df = alter_predicted_csv(predicted_bbox_df)

predicted_bbox_df.to_csv('./results/predicted.csv', index=False)


with open('./results/Images2Product.json', 'w') as file :

    json.dump(Images2Product, file)


print("\nImages2Product json dumped in './results/' folder")


# Compute the object wise confusion matrix using the pre-written script
cnf_matrix = compute_confison_matrix(predicted_bbox_df, ground_truth_df)


print("\nPrecision = ",cnf_matrix['precision'])
print("\nRecall = ",cnf_matrix['recall'])
print("\nAccuracy = ",cnf_matrix['acc'])


with open('./results/metrics.json', 'w') as file:
    
    json.dump(cnf_matrix, file)
    
    
print("\nMetrics json dumped in './results/' folder")




