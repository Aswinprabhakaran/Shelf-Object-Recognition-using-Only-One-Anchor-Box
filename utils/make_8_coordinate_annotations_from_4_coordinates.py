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


master_annotations = []


for line in annotations:
    
    frame_bb = []
    
    line = line.split(" ")
    
    frame_bb.append(line[0])
    
    for bb in line[1:]:
        
        bb = bb.split(",")
        
        xmin = bb[0]
        
        ymin = bb[1]
        
        xmax = bb[2]
        
        ymax = bb[3]
        
        some = [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, bb[-1]]
        
        frame_bb.append(",".join(some))
        
    master_annotations.append(" ".join(frame_bb))



assert len(master_annotations), len(annotations)


file_name = args['input_ann_file'].split("/")[-1]

file = open(args['output_path'] + 'converted_8_coordinate_{}'.format(file_name), 'w')

for line in master_annotations:
    
    file.write(line+"\n")
    
file.close()


print("\nConverted annotations saved to save path")
