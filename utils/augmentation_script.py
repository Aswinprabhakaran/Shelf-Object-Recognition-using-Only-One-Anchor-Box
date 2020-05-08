#!/usr/bin/env python

import cv2
import numpy as np
import random

# # Shearing
def shearing(image, bounding_box):
    
    shear_factor = random.sample([0.1,0.2,0.3,0.4,0.5],1)[0]
    
    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])
                
    nW =  image.shape[1] + abs(shear_factor * image.shape[0])
    
    if bounding_box.size != 0:
        
        bounding_box[:,[0, 2, 4, 6]] += ((bounding_box[:,[1, 3, 5, 7]])*abs(shear_factor))
    
    else:
        
        bounding_box = []
        
    image = cv2.warpAffine(image, M, (int(nW), image.shape[0]))

    return image , bounding_box


# # Rotate
def rotate_im(image, angle):
    
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def rotate_box(corners,angle,  cx, cy, h, w):
    
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def Rotate(image, bounding_box):
    
    angle = random.sample(list(np.arange(-25,26)),1)[0]
    
    w,h = image.shape[1], image.shape[0]
    
    cx, cy = w//2, h//2
        
    image = rotate_im(image, angle)
        
    bounding_box[:,:8] = rotate_box(bounding_box[:,:8], angle, cx, cy, h, w)
    
    return image , bounding_box


# # Translate
def translate(image , bounding_box):
    
    translate = random.sample(list((np.arange(-25,26)/100)),2)
    
    translate_factor_x = translate[0]
    translate_factor_y = translate[1]
    
    img_shape = image.shape
        
    #get the top-left corner co-ordinates of the shifted box 
    corner_x = int(translate_factor_x * img_shape[1])
    corner_y = int(translate_factor_y * img_shape[0])
    
    image = image[ max(-corner_y, 0) : min(img_shape[0], -corner_y + img_shape[0]),
                  max(-corner_x, 0) : min(img_shape[1], -corner_x + img_shape[1]),
                  :]
    
    image_shape = [max(-corner_y, 0) , min(img_shape[0], -corner_y + img_shape[0]),
                   max(-corner_x, 0) , min(img_shape[1], -corner_x + img_shape[1])]
    
    
    if (corner_x) < 0  and (corner_y < 0):
        
        bounding_box[:,:8] += [corner_x, corner_y,corner_x, corner_y, 
                               corner_x, corner_y,corner_x, corner_y]
        
    elif corner_x > 0 and corner_y < 0 :
        
        bounding_box[:,[1,3,5,7]] += [corner_y, corner_y, corner_y, corner_y]
        
    
    elif corner_x < 0 and corner_y > 0 :
        
        bounding_box[:,[0,2,4,6]] += [corner_x ,corner_x, corner_x, corner_x]
        
    
    final_box = []
    
    YMIN = image_shape[0]
    YMAX = image_shape[1] - YMIN
    
    XMIN = image_shape[2]
    XMAX = image_shape[3] - XMIN
   
    for index, box in enumerate(bounding_box):
        
        x_min = min(box[0], box[2], box[4] ,box[6])
        x_max = max(box[0], box[2], box[4] ,box[6])
        
        y_min = min(box[1], box[3], box[5] ,box[7])
        y_max = max(box[1], box[3], box[5] ,box[7])
        
        if x_max <= XMAX and y_max <= YMAX and x_min >= XMIN and y_min >= YMIN:
            
            final_box.append(box)
            
    
    return image, np.array(final_box)


# # Resize
def Resize(image, bounding_box):
    
    scale_percent = random.sample(list(np.arange(75,126)/100),1)[0] # percent of original size
    
    width = int(image.shape[1] * scale_percent )
    height = int(image.shape[0] * scale_percent )
    
    dim = (width, height)
    
    # resize image
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    
    
    if bounding_box.size != 0:
    
        bounding_box[:,:8] *= scale_percent
    
    else:
        
        bounding_box = []
        
    return resized_image, bounding_box


