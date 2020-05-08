#!/usr/bin/env python

import numpy as np
from ast import literal_eval
import argparse, os

ap = argparse.ArgumentParser()

ap.add_argument( '-a' , '--input_ann_file' , required = True , type = str , help = 'path to input augmented annotations.txt file' )

ap.add_argument( '-o' , '--output_path' , required = True , type = str , help = 'path to save the output error files' )

args = vars(ap.parse_args())

if not os.path.exists(args['output_path']):
    os.makedirs(args['output_path'])


with open(args['input_ann_file'],'r') as f:
    
    annotations = f.readlines()
    
f.close()

annotations = [ann.strip() for ann in annotations]

print("\nNo of augmented annotations in the file :",len(annotations))

def get_enclosing_box(corners):
    
    zeros = np.zeros(corners.shape, dtype=int)
    
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
    
    return final

def make_alterations(boxes):
    
    final_box = []
    
    boxes = boxes.astype(int)
    
    for box in boxes:
        
        box = box.tolist()
        
        box = [str(x) for x in box]
        
        final_box.append(",".join(box))
        
    return " ".join(final_box)


final_annotations = []

for index, ann in enumerate(annotations):
    
    ann = ann.split(" ")
    
    name = ann[0]
    
    bb_per_frame = []
    
    for bb in ann[1:]:
        
        bb = bb.split(",")
        
        bb = [literal_eval(x) for x in bb]
        
        bb_per_frame.append(bb)
        
    bb_per_frame = np.array(bb_per_frame)
    
    returned_box  = get_enclosing_box(bb_per_frame)
    
    ann_string = make_alterations(returned_box)  
    
    to_append = 'aug_images/'+name+" "+ann_string
    
    final_annotations.append(to_append)


print("\nAltered Annotations to yolo format :",len(final_annotations))


file = open(args['output_path'] + 'yolo_format_training_annotations.txt', 'w')

for line in final_annotations:
    
    file.write(line+"\n")
    
file.close()





