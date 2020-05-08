#!/usr/bin/env python

import os, cv2
import numpy as np
import ast
import argparse, os

ap = argparse.ArgumentParser()

ap.add_argument( '-a' , '--input_ann_file' , required = True , type = str , help = 'path to input augmented annotations.txt file' )

args = vars(ap.parse_args())

with open(args['input_ann_file'],'r') as f:
    
    annotations = f.readlines()
    
f.close()

annotations = [ann.strip() for ann in annotations]

print("\nNo of augmented annotations in the file :",len(annotations))

altered_annotations = []

for line in annotations:
    
    line = line.split(" ")
    
    this = [line[0]]
    
    boxes = [ast.literal_eval(x) for x in line[2:]]
    
    boxes = np.array(boxes).reshape(-1,5)
    
    for box in boxes:
        
        b_box = [box[0], box[1], box[0] + box[2], box[1] + box[3], 0]
        b_box = [str(x) for x in b_box]
        
        this.append(",".join(b_box))
        
    assert int(line[1]) == (len(this)-1), "original {}, we hvae {}".format(line[1], len(this)-1)
        
    altered_annotations.append(" ".join(this))


assert len(altered_annotations) == len(annotations)

save_path = "/".join(args['input_ann_file'].split("/")[:-1])

file =  open(os.path.join(save_path,'test_annotations_pascal_VOC_format.txt'), 'w')

for line in altered_annotations:
    
    file.write(line+"\n")
    
file.close()
