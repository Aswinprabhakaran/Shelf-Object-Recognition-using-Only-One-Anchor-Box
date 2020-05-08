#!/usr/bin/env python
import cv2
import numpy as np
np.random.bit_generator = np.random._bit_generator
import random
import imgaug.augmenters as iaa
import os
import ast
import augmentation_script as img_aug
import argparse

# # Augmentations

ap = argparse.ArgumentParser()

ap.add_argument('-img_path', '--images_path',required = True , type = str , help = 'path to images to augment')

ap.add_argument('-ann', '--annotations_path',required = True , type = str , help = 'path to annotations for the fgiven images')

ap.add_argument('-o', '--output_path',required = True , type = str , help = 'path to save the Augmented annotations and images')

args = vars(ap.parse_args())

print(args)

if not os.path.exists(args['output_path']):
    os.makedirs(os.path.join(args['output_path'], 'RTShear'))
    os.makedirs(os.path.join(args['output_path'], 'RTResize'))
    os.makedirs(os.path.join(args['output_path'], 'RTBlur'))
    os.makedirs(os.path.join(args['output_path'], 'RTNoise')) 
    os.makedirs(os.path.join(args['output_path'], 'RTSharp'))
    os.makedirs(os.path.join(args['output_path'], 'aug_images'))


# Functions to perform 3 types of augmentation on single image
def pipeline_RTShear(image, box):

    rotated_image , rotated_box = img_aug.Rotate(image.copy() , box.copy())
    translated_image, translated_box = img_aug.translate(rotated_image, rotated_box)
    sheared_image, sheared_box = img_aug.shearing(translated_image, translated_box)
    
    final_box = []
    
    if len(sheared_box) > 0:
    
        for box in sheared_box.tolist():
        
            box = [str(x) for x in box]
        
            final_box.append(",".join(box))
    
        final_box = " ".join(final_box)
    
    
    return sheared_image, final_box



def pipeline_RTResise(image, box):

    rotated_image , rotated_box = img_aug.Rotate(image.copy() , box.copy())
    translated_image, translated_box = img_aug.translate(rotated_image, rotated_box)
    Resize_image, Resize_box = img_aug.Resize(translated_image, translated_box)

    final_box = []
    
    if len(Resize_box) > 0: 

        for box in Resize_box.tolist():
        
            box = [str(x) for x in box]
        
            final_box.append(",".join(box))
    
        final_box = " ".join(final_box)
    
    return Resize_image, final_box


def pipeline_RTBlur(image, box):

    rotated_image , rotated_box = img_aug.Rotate(image.copy() , box.copy())
    translated_image, translated_box = img_aug.translate(rotated_image, rotated_box)
    
    blur_aug = iaa.MotionBlur(k = 10)
    
    blured_image = blur_aug(image = translated_image)
    
    final_box = []

    if len(translated_box) > 0 :

        for box in translated_box.tolist():
            
            box = [str(x) for x in box]
            
            final_box.append(",".join(box))
    
        final_box = " ".join(final_box)
    
    return blured_image, final_box


def pipeline_RTSharp(image, box):

    rotated_image , rotated_box = img_aug.Rotate(image.copy() , box.copy())
    translated_image, translated_box = img_aug.translate(rotated_image, rotated_box)
    
    sharp_aug = iaa.Sharpen(alpha = random.sample(list(np.arange(11)/10),2))
    
    sharped_image = sharp_aug( image = translated_image)
    
    final_box = []
    
    if len(translated_box) > 0 :

        for box in translated_box.tolist():
            
            box = [str(x) for x in box]
            
            final_box.append(",".join(box))
    
        final_box = " ".join(final_box)
    
    return sharped_image, final_box


def pipeline_RTNoise(image, box):

    rotated_image , rotated_box = img_aug.Rotate(image.copy() , box.copy())
    translated_image, translated_box = img_aug.translate(rotated_image, rotated_box)
    
    noise_aug = iaa.AdditiveGaussianNoise(loc = (0.0, 0.1*255),
                                      scale=(0.0, 0.1*255),
                                      per_channel = True)
    
    noised_image = noise_aug( image = translated_image)
    
    final_box = []

    if len(translated_box) > 0 :
        
        for box in translated_box.tolist():
            
            box = [str(x) for x in box]
            
            final_box.append(",".join(box))
    
        final_box = " ".join(final_box)
    
    return noised_image, final_box


# Reading the images form the images path
images_to_aug = os.listdir(args['images_path'])

print("\nNo of Images to Augmentations :", len(images_to_aug))

# Reading the annotaions from the annotations.txt
with open(args['annotations_path'], 'r') as f:
    
    annotations = f.readlines()
    
f.close()

annotations = [ann.strip() for ann in annotations]

print("\nNo of Annotations in the given annotations file :",len(annotations))

annotations_dict = {}

for ann in annotations:
    
    ann = ann.split(" ")
    
    full_box = []
    
    for box in ann[1:]:
        
        box = box.split(",")
        
        box = [ast.literal_eval(x) for x in box]
        
        full_box.append(box)
    
    annotations_dict[ann[0]] = np.array(full_box, dtype = np.float64)
    
del annotations

# To save final augmented annotations
augmentated_annotations = []

for index, img in enumerate(sorted(images_to_aug)):
    
    image = cv2.imread(args['images_path']+img)
    
    bounding_box = annotations_dict[img]
    
    p1_img, p1_box = pipeline_RTShear(image =  image.copy(), box = bounding_box.copy())
    
    if p1_box:
    
        cv2.imwrite('{}/RTShear/RTShear_{}'.format(args['output_path'], img), p1_img)
        
        cv2.imwrite('{}/aug_images/RTShear_{}'.format(args['output_path'], img), p1_img)
    
        augmentated_annotations.append('RTShear_'+img+" "+p1_box)
        
    p2_img, p2_box = pipeline_RTResise(image.copy(), bounding_box.copy())
    
    if p2_box:
    
        cv2.imwrite('{}/RTResize/RTResize_{}'.format(args['output_path'], img), p2_img)
        
        cv2.imwrite('{}/aug_images/RTResize_{}'.format(args['output_path'], img), p2_img)
    
        augmentated_annotations.append('RTResize_'+img+" "+p2_box)
        
        
    p3_img, p3_box = pipeline_RTBlur(image =  image.copy(), box = bounding_box.copy())
    
    if p3_box:
    
        cv2.imwrite('{}/RTBlur/RTBlur_{}'.format(args['output_path'], img), p3_img)
        
        cv2.imwrite('{}/aug_images/RTBlur_{}'.format(args['output_path'], img), p3_img)
    
        augmentated_annotations.append('RTBlur_'+img+" "+p3_box)
        
    p4_img, p4_box = pipeline_RTNoise(image =  image.copy(), box = bounding_box.copy())
    
    if p4_box:
    
        cv2.imwrite('{}/RTNoise/RTNoise_{}'.format(args['output_path'], img), p4_img)
        
        cv2.imwrite('{}/aug_images/RTNoise_{}'.format(args['output_path'], img), p4_img)
    
        augmentated_annotations.append('RTNoise_'+img+" "+p4_box)
        
    p5_img, p5_box = pipeline_RTSharp(image =  image.copy(), box = bounding_box.copy())
    
    if p5_box:
    
        cv2.imwrite('{}/RTSharp/RTSharp_{}'.format(args['output_path'], img), p5_img)
        
        cv2.imwrite('{}/aug_images/RTSharp_{}'.format(args['output_path'], img), p5_img)
    
        augmentated_annotations.append('RTSharp_'+img+" "+p5_box)
        
        
    if index % 10 == 0:
        
        print("Processed = ", index)


file = open('{}/augmentated_annotations.txt'.format(args['output_path']), 'w')

for line in augmentated_annotations:
    
    file.write(line+"\n")
    
file.close()

print("\nTotal no of images augmented are {} from {}".format(len(augmentated_annotations), len(images_to_aug)))
