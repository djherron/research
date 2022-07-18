#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 20:44:27 2022

@author: dave
"""

'''
This script uses a Faster RCNN ResNet50-FPN object detection model trained
on VRD dataset images using visual relationship annotations customised by
DH to perform inference on VRD dataset images.  The predictions (inferences)
of the model (bboxes, labels, confidence scores) are then stored (along with
the names and sizes of the associated images) in a dictionary. The 
dictionary is then written to a JSON file.

The design of this script is based on the Jupyter Notebook named
'vrd_frcnn_inference_1.ipynb'. That notebook was designed to do inference
on one image at a time and to display the image along with the predicted
bboxes and class labels to facilitate visual evaluation of the performance
of the object detection.  That notebook was designed to be executed on a
CPU of a local machine.

This script is designed to run remotely, on a server cluster, in order to
take advantage of a GPU. It automates the inference process on a Test Set
worth of images. There is no provision for visual inspection of the 
inference results.
'''

#%%

import torch
import torchvision

from vrd_dataset_frcnn import VRDDataset

import json
import os

#%%

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'compute device being used: {device}')

#%% set the file path root

if os.getcwd().endswith('objectDetection'): 
    filepath_root = '..'
else:
    filepath_root = '.'

#%% manage the VRD object classes

# set path to VRD object class names 
vrd_obj_file = os.path.join(filepath_root, 'data', 'annotations_customised',
                            'vrd_dh_objects.json')

# get the VRD object class names 
with open(vrd_obj_file, 'r') as file:
    vrd_object_class_names = json.load(file)

# The object classes of the VRD dataset do not include a 'background' class.
# A Faster R-CNN object detection model expects needs a 'background' class 
# (at index 0). So to customise our pre-trained Faster R-CNN model for
# the VRD dataset correctly, we need to increment the number of object
# classes by 1.

print(f"Nr of VRD object classes (excl. 'background'): {len(vrd_object_class_names)}")

n_vrd_object_classes = len(vrd_object_class_names) + 1

#%% set the VRD dataset image directory

vrd_img_dir = os.path.join(filepath_root, 'data', 'train_images')

print(f'Image dir: {vrd_img_dir}')

#%% set the VRD dataset image annotations file path

#vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations_customised', 
#                             'vrd_dh_annotations_train_testsubset.json')

vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations_customised', 
                             'vrd_dh_annotations_train_trainsubset.json')

print(f'Annotations file: {vrd_anno_file}')

#%% prepare the VRD dataset image names

# load the annotations for the VRD dataset images for which we are
# about to perform object detection inference
with open(vrd_anno_file, 'r') as file:
    vrd_img_anno = json.load(file)

# extract the image names from the annotations
vrd_img_names = list(vrd_img_anno.keys())

print(f'Number of VRD images to be processed: {len(vrd_img_names)}')

#%% Load a pre-trained Faster R-CNN ResNet-50 FPN model

# set model hyper-parameters
model_args = {'box_score_thresh': 0.75,
              'num_classes': 91,
              'rpn_batch_size_per_image': 256,
              'box_batch_size_per_image': 256}

# instantiate a basic model loaded with pre-trained parameters
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,
                                                             **model_args)

print('Faster R-CNN model instantiated and loaded with pre-trained parameters')

#%% Customise the model for the VRD dataset

# adjust the number of output features in the classification output layer (for
# the object class predictions) to match the number of object classes in the 
# VRD dataset (plus 1 for a 'background' class)
model.roi_heads.box_predictor.cls_score.out_features = n_vrd_object_classes

# adjust the number of output features in the regression output layer (for the
# bounding box predictions) to match 4 times the number of object classes in 
# the VRD dataset (plus 1 for a 'background' class)
model.roi_heads.box_predictor.bbox_pred.out_features = n_vrd_object_classes * 4

# replace the classification output layer's weights and biases with new random ones 
# whose dimensions match the layer's revised output size
in_feat = model.roi_heads.box_predictor.cls_score.in_features
out_feat = model.roi_heads.box_predictor.cls_score.out_features
weights = torch.rand(out_feat, in_feat)
model.roi_heads.box_predictor.cls_score.weight = torch.nn.Parameter(weights, requires_grad=True)
biases = torch.rand(out_feat)
model.roi_heads.box_predictor.cls_score.bias = torch.nn.Parameter(biases, requires_grad=True)

# replace the bbox regression output layer's weights and biases with new random ones 
# whose dimensions match the layer's revised output size
in_feat = model.roi_heads.box_predictor.bbox_pred.in_features
out_feat = model.roi_heads.box_predictor.bbox_pred.out_features
weights = torch.rand(out_feat, in_feat)
model.roi_heads.box_predictor.bbox_pred.weight = torch.nn.Parameter(weights, requires_grad=True)
biases = torch.rand(out_feat)
model.roi_heads.box_predictor.bbox_pred.bias = torch.nn.Parameter(biases, requires_grad=True)

print('Faster R-CNN model customised for VRD dataset')

#%% function to load trained model checkpoint file

def load_model_checkpoint(checkpoint_path, model):
    
    # load the model checkpoint file
    model_checkpoint = torch.load(checkpoint_path, 
                                  map_location=torch.device('cpu'))
    
    # load the parameters representing our trained object detector 
    # into our (customised) model instance
    model.load_state_dict(model_checkpoint['model_state_dict'])

    # get other variables saved to the checkpoint file
    epoch = model_checkpoint['epoch']
    avg_loss_per_mb = model_checkpoint['avg_loss_per_mb']
    
    print(f"Model checkpoint file loaded: {checkpoint_path}")
    
    return epoch, avg_loss_per_mb

#%% load a checkpoint file representing our trained object detector model

# loading a checkpoint file overwrites all of the weight and bias parameters
# of the pre-trained (and customised) model with those of our trained
# object detector model

checkpoint_path = 'vrd_frcnn_v2_checkpoint_510.pth'
args = {'checkpoint_path': checkpoint_path, 'model': model}
last_epoch, last_avg_loss_per_mb = load_model_checkpoint(**args)

print(f"last epoch: {last_epoch}; last avg loss per mb: {last_avg_loss_per_mb:.4f}")

#%% prepare our VRD Dataset object for loading images and target annotations

# instantiate our VRD Dataset class
dataset_item = VRDDataset(vrd_img_dir=vrd_img_dir, vrd_anno_file=vrd_anno_file)

print('VRD Dataset class instantiated')

# Note:
# when called, a VRDDataset object returns a 3-tuple: (idx, img, targets)
# idx - the index of an image's entry within the dataset annotations
# img - an image (formatted for the Faster RCNN model)
# targets - a dictionary with keys 'boxes' and 'labels' (formatted for
#           the Faster RCNN model); during inference, however, the targets
#           are NOT passed into the model

#%% push the model to the GPU (if it's available)

if device == torch.device('cuda'):
    model = model.to(device)

#%% put the model in evaluation (inference) mode

model.eval()

print('Model placed on correct device and put into evaluation mode')

#%% main processing loop

print('Main processing loop begins ...')

cnt = 0

results = {}

for imname in vrd_img_names:
    
    cnt += 1
    if cnt % 1 == 0:
        print(f'processing image {cnt}')
    
    idx = vrd_img_names.index(imname)
    
    idx2, img, targets = dataset_item[idx]
    
    if idx2 != idx:
        raise ValueError('Problem: idx2 {idx2} != idx {idx} for image {imname}')
    
    img_height = img.size()[1]
    img_width = img.size()[2]
    print(f'img {imname}: size {img.size()}')

    out = model([img])
    
    # convert the outputs from tensors to numpy arrays
    boxes = out[0]['boxes'].detach().numpy()
    labels = out[0]['labels'].numpy()
    scores = out[0]['scores'].detach().numpy()
    
    # store the results in the results dictionary
    results[imname] = {'imwidth': img_width,
                       'imheight': img_height,
                       'boxes': boxes,
                       'labels': labels,
                       'scores': scores}

    if cnt > 9:
        break 

print('Main processing loop complete')
    
#%%

print()
print('Displaying results')
print()
for k,v in results.items():
    print(k,v)
    print()

#%% save results dictionary to JSON file

filename = 'vrd-frcnn-v2-inference-results-1.json'
with open(filename, 'w') as file:
    json.dump(results, file)

print()
print(f'Results saved to file {filename}')

#%%

print()
print('Processing complete')





















