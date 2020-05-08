# Necessary Imports

import os
import pandas as pd
import math
from shapely.geometry import Polygon
import numpy as np


def make_polygon_from_4_coordinates(box):
    
    # Create a polygon of 8 coordinates from 4 coordinates of a rectangle assuming the 
    # box to be given in xmin , ymin , xmax, ymax
    
#     return Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)])
    
    return Polygon([ (box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3]) ])


def make_polygon_from_8_coordinates(box):
    
    # Create a polygon of 8 coordinates assuming the box to be given in 
    # x1, y1, x2, y2, x3, y3, x4, y4
        
#     return Polygon([(x1,y1),(x2,y2),(x3,y3),(x4,y4)])

    return Polygon([(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])])
    
    
def centroid_distance(box): 
    
    x1 = box[0]
    y1 = box[1]
    
    x2 = box[2]
    y2 = box[3]
    
    # Computes distance between two coordinates 

    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  


def get_centroid_from_8_coordinates(box):
    
    xmin = min(box[0],box[2], box[4], box[6])
    xmax = max(box[0],box[2], box[4], box[6])

    ymin = min(box[1], box[3], box[5], box[7])
    ymax = max(box[1], box[3], box[5], box[7])
    
    CX = ( xmin + xmax ) / 2
    CY = ( ymin + ymax ) / 2
    
    return CX, CY


def get_centroid_from_4_coordinates(box):
    
    CX = ( box[0] + box[2] ) / 2
    CY = ( box[1] + box[3] ) / 2
    
    return CX, CY

def alter_predicted_csv(predicted_df):
    
    n = (len(predicted_df.columns) - 1) // 4

    col = []

    col.append("name")
    
    for i in range(1,n+1):
        xmin='xmin'+ str(i)
        ymin='ymin'+ str(i)
        xmax='xmax'+ str(i)
        ymax='ymax'+ str(i)
        col.append(xmin)
        col.append(ymin)
        col.append(xmax)
        col.append(ymax)
    

    predicted_df.columns = col
    
    predicted_df.sort_values(by=['name'], inplace=True)
    
    return predicted_df



def compute_confison_matrix(predicted_df, ground_truth_df, iou_intersection_th = 0.4 ,debug = False,
                            calculate_distance = False, minimum_distance = None):
    
    """
    
    This piece of code essentially computes the confusion matrix between Ground Truth and predicted Video
    
    :param predicted_df -- prediction dataframe
    :param ground_truth_df -- groundtruth dataframe
    :param iou_intersection_th -- IOU iou_intersection_th threshold
    :param calculate_distance -- If True , compute the centroid distance between true and predicted boxes
    :param minimum_distance -- Distance threshold between two pred and true boxes
    
    """    
    
    # Defining the Output variables which has to be returned
    result = {}
    
    # Defining the variables which wil be used in the computation
    tp = 0
    tn = 0
    unique_all_gt_box = []
    unique_all_pred_box= []
    
    
    df_pred = predicted_df.copy()
    
#     df_pred = alter_predicted_csv(predicted_df = df_pred)

    GT = ground_truth_df.copy()
    
    # Setting up an dataframe to store the details of matched predicted box with GroundTruth Box 
    count_correctness = pd.DataFrame(columns=['NAME','IOU_THRESHOLD', 'GT_BOX', 
                                           'MATCHED_PRED_BOX', 
                                           'IOU_BTWN_GT_AND_PRED_BOX',
                                           'COUNT_OF_OTHER_BOXES_MATCHED_WITH_GT_FOR_GIVEN_IOU'])


    # Processing the frames in the csv to compute its confusion matrix
    for frame_name in GT['name'].unique().tolist():
    
        # Values changing w.r.t. every frame in the given video file
        bbox_num_mapping = {}
        bbox_with_iou = {}
        matched_pred_box = {}
        
        # Getting all the GroundTruth Frames for the given FRAME NAME or FRAME NO from GroundTruth dataframe
        cur_frame_gt_boxes = GT[GT['name'] == frame_name].values.tolist()
        
        for box in cur_frame_gt_boxes:
        
            bbox_num_mapping[box[1]] = box[2:10]
        
            bbox_with_iou [box[1]] = 0.0
        
            unique_all_gt_box.append(box[2:10])
         
        
        # Getting all the prediction for the current frame from the predictions csv
        cur_frame_predictions =  df_pred[df_pred['name'] == frame_name].values.tolist()
        
        if cur_frame_predictions:
        
            cur_frame_pred_boxes = np.array(cur_frame_predictions[0][1:]).reshape(-1,4).tolist()

            cur_frame_pred_boxes = [box for box in cur_frame_pred_boxes if not pd.isna(box[0])]
        
            for pred_box in cur_frame_pred_boxes :
            
                unique_all_pred_box.append(pred_box)
            
                pred_box_poly = make_polygon_from_4_coordinates(box = pred_box)
            
                per_pred_box_iou = {}
            
                for box_no , gt_box in bbox_num_mapping.items():
                    
                    gt_box_poly = make_polygon_from_8_coordinates(box = gt_box)
                
                    iou = gt_box_poly.intersection(pred_box_poly).area / gt_box_poly.union(pred_box_poly).area
                
                    if calculate_distance:
                    
                        x_centroid, y_centroid = get_centroid(box = gt_box)
                        x_centroid_p, y_centroid_p = get_centroid_from_4_coordinates(box = pred_box)
                    
                        centroid_dis = centroid_distance(x_centroid, y_centroid, x_centroid_p, y_centroid_p)
                    
                        if (iou >= iou_intersection_th and centroid_dis < minimum_distance):
                        
                            per_pred_box_iou[box_no] = iou
                
                    else:
                    
                        if (iou >= iou_intersection_th):
                        
                            per_pred_box_iou[box_no] = iou
                
            
                if per_pred_box_iou:
                
                    matched_gt_box_no = max(per_pred_box_iou, key = per_pred_box_iou.get)
            
                    if per_pred_box_iou[matched_gt_box_no] > bbox_with_iou[matched_gt_box_no]:
                
                        bbox_with_iou[matched_gt_box_no] = per_pred_box_iou[matched_gt_box_no]
                
                        matched_pred_box[matched_gt_box_no] = pred_box
                
        
        tp += len(matched_pred_box)
        
        for gt_box_no , mat_box in matched_pred_box.items():
        
            count_correctness = count_correctness.append({'NAME' : frame_name,
                                                          'IOU_THRESHOLD' : iou_intersection_th,
                                                          'GT_BOX' : bbox_num_mapping[gt_box_no],
                                                          'MATCHED_PRED_BOX' : mat_box,
                                                          'IOU_BTWN_GT_AND_PRED_BOX' : bbox_with_iou[gt_box_no]}, ignore_index = True)
        
    
    # False Positives - unique all predicted boxes - true predicted boxes ( TRUE POSITIVES)
    fp = len(unique_all_pred_box) - tp
    
    assert len(GT) == len(unique_all_gt_box)
    
    # False Negatives - unique all GroundTruth boxes - true predicted boxes ( TRUE POSITIVES)
    fn = len(unique_all_gt_box) - tp
    
    
    # Computation of Confusion Matrix
    if tp == 0:
        acc = 0
        recall = 0
        precision = 0
        f_measure = 0
    else:
        acc = (tp+tn)/(tp+tn+fp+fn) 
        recall = tp/(tp+fn)
        precision = tp/(tp+fp)
        f_measure = (2*recall*precision) / (recall+precision)
    
    # Assigning the results to the 
    result['tp'] = tp
    result['fp'] = fp
    result['fn'] = fn
    result['tn'] = tn
    result['acc'] = acc * 100
    result['precision'] = precision
    result['recall'] = recall
    result['f_measure'] = f_measure
    
    if debug:
        
        result['count_correctness'] = count_correctness
        
    return result
