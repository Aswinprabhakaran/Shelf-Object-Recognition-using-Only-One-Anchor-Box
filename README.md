# Keras - Single Shot Object Detection

## Introduction

tf.keras implementation of a Single Shot Object Detector Architecture with different technologies support:

#### Backbone
- [x] Darknet53/Tiny Darknet

#### Head
- [x] Tiny YOLOv3

#### Loss
- [x] Standard YOLOv3 loss

#### Postprocess
- [x] Numpy YOLOv3 postprocess implementation
- [x] TFLite/MNN C++ YOLOv3 postprocess implementation
- [x] TF YOLOv3 postprocess model
- [x] tf.keras batch-wise YOLOv3 postprocess Lambda layer

#### Train tech
- [x] Transfer training from Darknet
- [x] Singlescale image input training
- [x] Multiscale image input training
- [x] Cosine learning rate decay
- [x] Pruned model training (only valid for TF 1.x)


## Quick Start

1. Install the requirements:

```
# pip install -r requirements.txt

```

1.1 If you are using anachonda or miniconda , plz create the env with the provided environment file

```
# conda env create -f environment.yml

```

4. Run inference on provided test images

```

# python inference.py

```


## Guide of docs and scripts


### utils :

This folder contains scripts for generating augmented data , converting annotation form COCO format to PASCAL-VOC format(yolo acceptable format) and few other scripts like plotting the 
annotation on the images to ensure the hygiene practises in computer vision which gives us visual correctness of annotations.

This folder also contains script to generate 8 coordinate annotations from 4 coordinate (pascal-VOC) format annotation.

Since the data augmentation script takes 8 coordinate annotation , it is necesaary to convert the annotations to 8 coordinate.

### Training

Problem was to train a Single Shot Object detector with only one anchor box per feature-cell map.

Different Single Shot Object detector :

1. SSD
2. YOLO


Given a data of 354 images comprised of different angles. Made the split as 283 images for training and 71 images for testing.

As the trainning data size is too small for a deep learning model to learn, performed 'Augmentation' to increase the data size.

[Augmentation](https://github.com/Aswinprabhakaran/product_detection_aswin_prabhakaran/blob/master/utils/augmentation_script.py) - performs 5X augmentation on a single image and thus genearted 1415 images for training.
 
Augmented Training Image are in './aug_images/' path


*** Tiny Yolo ***


As the data size is reasonabel to train a model, trained a tiny yolo wth keras implementation.

train.py -- provides the support for the training multiple SSOD models, used one such feature to train my model.

Did transfer learning by loading the pretrained tiny yolo weights from [YOLO website](http://pjreddie.com/darknet/yolo/) and [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet).




This super cool repo supports two phases of tranfer learning training in a easy way :

1. Phase 1 - unfreeze only last layer of loaded pretrained model or unfreeze backbone of the model or unfreeze all

this can be passed as a parameter while training the model and transfer training can be done

2. Phase 2 - Unfreeze all the layers and begin real training for the data supplied.

This way i trained my "Tiny YOLO" for 100 epochs to get a reasonal validation loss. 


### Inference 

```

# python inference.py

```

Post training since some pre-processing is required , uses keras - post-processing to get the bounding box and class scores.

the above python scripts runs the prediction on the test images in the directory and saves then in the results folder.

So once the detection is done , next comes the evaluation.

### Evaluation


Since this is a task of object detection , i have computed the precision and recall between the no of objects predicted and no of objects in the groundtruth.

confusion_matrix.py -- This scripts takes care of computing the confusion matrix.

Once the inference is done, the script calls the matrix and make the result.

So all we have to do is just run :

```
# python inference.py

```

### Results 

This folder contains :

1. The image with predicted bounding boxes plotted on them with confidence score of the box.
2. Images2Products.json - representing the mapping of no of productes identidfied by the model for the given image @ a confidence greater than 0.3 and NMS of 0.4.
3. metrics.json - Contains the precision , recall and others values computed between groundtruth and predicted with an IOU of 0.4 between GT_boxes and pred_boxes
4. Predicted_csv - Image wise predicted bounding boxes


Make sure that you replicate the environment before testing



### Questions asked:

1. What is the purpose of using multiple anchors per feature map cell? 

This takes me to the what are anchors ?

Basically anchors are just fixed initial boundary box guesses.
SSD uses a matching phase while training, to match the appropriate anchor box with the bounding boxes of each ground truth object within an image. Essentially, the anchor box with the highest degree of overlap with an object is responsible for predicting that object’s class and its location. This property is used for training the network and for predicting the detected objects and their locations once the network has been trained. In practice, each anchor box is specified by an aspect ratio and a zoom level.

Back to the original question : 

Multiple anchors are used to identify/detect multiple objects of different shapes in one grid cell. 

2. Does this problem require multiple anchors? Please justify your answer.

Here we are focused on detecting only one type of object(a producy of 'X' company) where all the products are of similar sizes.

Though all the products seem to be in same size , i would suggest to use mutltie anchors because when applicating consequite convolution to the image , 
the spatial resolution of the image decreases which makes the model to easily identiy objects of larer size and to miss objects of small size.

Since the objects in this problem are of all smaller in size compared to the object size of a TV or a person or a car , etc., I would encourage to use multile anchors boxes of appropriate Width and height to get a better performance model.









  
