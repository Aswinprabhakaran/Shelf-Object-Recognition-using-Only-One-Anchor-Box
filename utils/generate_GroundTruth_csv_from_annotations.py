#!/usr/bin/env python

import numpy as np
import ast
import numpy as np
import pandas as pd
import argparse, os
import math

ap = argparse.ArgumentParser()

ap.add_argument( '-a' , '--input_ann_file' , required = True , type = str , help = 'path to input annotations.txt file' )

ap.add_argument( '-o' , '--output_path' , required = True , type = str , help = 'path to save the output error files' )

args = vars(ap.parse_args())

if not os.path.exists(args['output_path']):
    os.makedirs(args['output_path'])


with open(args['input_ann_file'],'r') as f:
    
    annotations = f.readlines()
    
f.close()

annotations = [ann.strip() for ann in annotations]
annotations = [ann.replace(" ", ",") for ann in annotations]
annotations = [ann.split(",") for ann in annotations]

ann_df = pd.DataFrame(data = annotations)

no_of_boxes = (len(ann_df.columns)-1) // 9

col = ['name']

for i in range(1, no_of_boxes+1):
    
    x1='x1'+ str(i)
    y1='y1'+ str(i)
    
    x2='x2'+ str(i)
    y2='y2'+ str(i)
    
    x3='x3'+ str(i)
    y3='y3'+ str(i)
    
    x4='x4'+ str(i)
    y4='y4'+ str(i)
    
    class_name = 'class'+str(i)
    
    col.append(x1)
    col.append(y1)
    
    col.append(x2)
    col.append(y2)
    
    col.append(x3)
    col.append(y3)
    
    col.append(x4)
    col.append(y4)
    
    col.append(class_name)


ann_df.columns = col

ann_df.sort_values(by='name', inplace = True)
ann_df.reset_index(inplace=True)
ann_df.drop(columns=['index'], inplace=True)

for col in ann_df.columns.to_list()[1:]:
    
    ann_df[col] = ann_df[col].astype(float)


df_new = pd.DataFrame(columns=['name','bounding_box_no','x1','y1','x2','y2','x3','y3','x4','y4','class'])

for index, row in ann_df.iterrows():
   
    name = row['name']  
    
    for o in range(1,no_of_boxes+1): 
        
        x1 = 'x1'+ str(o)
        y1 = 'y1'+ str(o)
        
        x2 = 'x2'+ str(o)
        y2 = 'y2'+ str(o)
        
        x3 = 'x3'+ str(o)
        y3 = 'y3'+ str(o)
        
        x4 = 'x4'+ str(o)
        y4 = 'y4'+ str(o)
        
        class_name = 'class'+str(o)

        df_new = df_new.append({'name':name,
                                'bounding_box_no': o,
                                'x1' : row[x1],
                                'y1' : row[y1],
                                'x2' : row[x2],
                                'y2' : row[y2],
                                'x3' : row[x3],
                                'y3' : row[y3],
                                'x4' : row[x4],
                                'y4' : row[y4],
                                'class' : row[class_name]}, ignore_index=True)  


df_new.dropna(axis = 0, inplace = True)

df_new.sort_values(by=['name','bounding_box_no'],  inplace=True)

df_new.to_csv(args['output_path']+'GT_TEST.csv', index = False)


print("\nTEST CSV MADEE FROM PASSED ANNPOTATIONS.TXT")


