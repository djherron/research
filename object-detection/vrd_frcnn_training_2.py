#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 15 18:32:00 2022

@author: dave
"""

'''
Object detection on the customised VRD dataset.

This is a script version of Jupyter Notebook 'vrd_frcnn_2.ipynb', designed
to run on a remove server like City's Camber server cluster.

This script trains the PyTorch Torchvision implementation of a
faster-rcnn-resnet50-fpn object detection model on the VRD
(visual relationship detection) dataset whose annotations have been 
heavily customised (quality-improved / de-noised) by Dave Herron.

We start with a Torchvision Faster R-CNN ResNet50 FPN model pre-trained
on the COCO train2017 datatset. The pre-trained model is trained to detect
only the first 91 (1 'background' class, plus 90 object classes) of the
full 183 object classes in the COCO 2017 dataset. We adapt the dual
(classification and bbox regression) output layers to fit our customised
VRD dataset and then train the model further on the customised VRD dataset.
'''

#%%

import torch
import torchvision

import os
import json
import time

from vrd_dataset_frcnn import VRDDataset

#%%

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
print(f'Device: {device}')

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

#%%

# set the VRD dataset image annotations file path
vrd_anno_file = os.path.join(filepath_root, 'data', 'annotations_customised', 
                             'vrd_dh_annotations_train_trainsubset.json')

print(f'Annotations file: {vrd_anno_file}')

# load the annotations for the VRD dataset images for which we are
# about to perform object detection inference
with open(vrd_anno_file, 'r') as file:
    vrd_img_anno = json.load(file)

# extract the image names from the annotations
vrd_img_names = list(vrd_img_anno.keys())

print(f'Number of VRD images: {len(vrd_img_names)}')

#%%

# Load a pre-trained Faster R-CNN ResNet50 FPN model

# Definitions of selected model hyper-parameters:
# box_score_thresh : only return proposals with score > thresh (inference only)
# num_classes : nr of object classes (including a 'background' class at index 0)
# rpn_batch_size_per_image : nr anchors sampled during training of the RPN, re RPN loss calc
# box_batch_size_per_image : nr proposals sampled during training of ROI classification head

# set model hyper-parameters
# num_classes must be 91 for the pre-trained model
model_args = {'box_score_thresh': 0.75,
              'num_classes': 91,
              'rpn_batch_size_per_image': 256,
              'box_batch_size_per_image': 256}

# load the pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                             **model_args)
print('Pretrained model loaded')

#%%  Customise the pre-trained model for the customised VRD dataset

#%%

# adjust the number of output features in the classification output layer (for the
# object class predictions) to match the number of object classes in the VRD dataset
# (plus 1 for a 'background' class)
model.roi_heads.box_predictor.cls_score.out_features = n_vrd_object_classes

# adjust the number of output features in the regression output layer (for the
# bounding box predictions) to match 4 times the number of object classes in the 
# VRD dataset (plus 1 for a 'background' class)
model.roi_heads.box_predictor.bbox_pred.out_features = n_vrd_object_classes * 4

#%%

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

print('Pretrained model customised for VRD dataset')

#%% Prepare for training

#%%

# define a custom 'collate()' function to pass to our Dataloader so that it 
# won't complain about the input images being of different size and throw a
# runtime Exception, which would otherwise force us to train with a batch-size of 1
#
# sources:
# https://discuss.pytorch.org/t/torchvision-and-dataloader-different-images-shapes/41026/3
# https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2

def frcnn_collate(batch):
    '''
    Arguments:
       batch : a list of (idx, img, targets) tuples created by calling VRDDataset[idx]
    '''
    idxs = [item[0] for item in batch]
    imgs = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    return idxs, imgs, targets

#%%

# instantiate our VRD Dataset class
dataset_item = VRDDataset(vrd_img_dir=vrd_img_dir, vrd_anno_file=vrd_anno_file)

# set batch size
# Camber server cluster GPUs (eg Nvidia Titan V) can cope with a 
# batch size of 4
# Hyperion HPC GPUs (Nvidia A100) can cope with a batch size of 8, and
# perhaps more
batch_size = 8
print(f'Batch size: {batch_size}')

# prepare a Dataloader
args = {'batch_size': batch_size, 'shuffle': True, 'collate_fn': frcnn_collate}
dataloader = torch.utils.data.DataLoader(dataset_item, **args)

print(f'Number of minibatches per epoch: {len(dataloader)}')

#%%

# put model in training mode
model.train()

#%%

# push model to GPU
if device == torch.device('cuda'):
    model = model.to(device)

#%%

# establish an optimiser
# (note: we do this AFTER having pushed our model to the GPU, which is a 
#  recommended convention; it's required especially when the optimiser maintains  
#  internal state, like the Adagrad optimiser)
opt_args = {'lr': 1e-5, 'weight_decay': 1e-3}
optimiser = torch.optim.Adam(model.parameters(), **opt_args)

#%%

def save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb):
    checkpoint = {'epoch': epoch, 
                  'model_state_dict': model.state_dict(),
                  'optimiser_state_dict': optimiser.state_dict(),
                  'avg_loss_per_mb': avg_loss_per_mb}
    filename = "vrd_frcnn_v2_checkpoint_" + str(epoch) + ".pth"
    torch.save(checkpoint, filename)
    print(f"Model checkpoint file saved: {filename}")
    
    return None

#%%

def load_model_checkpoint(filename, model, optimiser):
    # load checkpoint file; note that we don't set parameter map_location=device, 
    # which indicates the location where ALL tensors should be loaded;
    # this is because we want our model on the GPU but our optimiser on the CPU,
    # which is where they are both saved from when torch.save() is executed
    checkpoint = torch.load(filename)
    
    # initialise the model and optimiser state (in-place) to the saved state 
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    
    # get other saved variables
    epoch = checkpoint['epoch']
    avg_loss_per_mb = checkpoint['avg_loss_per_mb']
    
    print(f"Model checkpoint file loaded: {filename}")
    
    return epoch, avg_loss_per_mb

#%% Configure training loop parameters

# set training mode: 'start' or 'continue'
# - 'start' : no checkpoint file to load; begin training from epoch 0
# - 'continue' : load checkpoint file and resume training from last epoch
training_mode = 'continue'
continue_from_epoch = 800

# set number of training epochs
n_epochs_train = 50

# set frequency for saving checkpoint files (in number of epochs)
n_epochs_checkpoint = 10

# initialise training epoch range
if training_mode == 'start':
    last_epoch = 0
elif training_mode == 'continue':
    filename = "vrd_frcnn_v2_checkpoint_" + str(continue_from_epoch) + ".pth" 
    args = {'filename': filename, 'model': model, 'optimiser': optimiser}
    last_epoch, last_avg_loss_per_mb_over_epoch = load_model_checkpoint(**args)
    print(f"last epoch: {last_epoch}; last avg loss per mb over epoch: {last_avg_loss_per_mb_over_epoch:.4f}")
else:
    raise ValueError('training mode not recognised')

# initialise range of epoch numbers
first_epoch = last_epoch + 1
final_epoch = first_epoch + n_epochs_train

# minibatch group size
#mb_group_size = 10

#%%

print(f'We are using {torch.cuda.device_count()} GPU(s)')

# TODO: use of multiple GPUs is an unsolved problem; this code block
#       here is not sufficient by itself; we get run-time errors later
# if using multiple GPUs, prepare the model for parallelisation
#if torch.cuda.device_count() > 1:
#    print('About to prepare model for GPU parallelisation with DataParallel')
#    model = torch.nn.DataParallel(model)
#    print('Model prepared for GPU parallelisation with DataParallel')

#%% Training loop

start_time = time.time()
start_time_checkpoint = start_time

for epoch in range(first_epoch, final_epoch):
    
    print(f'\nepoch {epoch} starting ...')
    
    epoch_loss = 0
    #mb_group_loss = 0
    
    for bidx, batch in enumerate(dataloader):
        
        # split the batch into its 3 components
        idxs, images, targets = batch
        
        # push the training data to GPU
        if device == torch.device('cuda'):
            images_gpu = []
            targets_gpu = []
            for i in range(len(images)):
                image_gpu = images[i].to(device)
                images_gpu.append(image_gpu)
                target_gpu = {}
                target_gpu['labels'] = targets[i]['labels'].to(device)
                target_gpu['boxes'] = targets[i]['boxes'].to(device)
                targets_gpu.append(target_gpu)
            images = images_gpu
            targets = targets_gpu
        
        # forward pass through model
        loss_components = model(images, targets)
        
        # sum the 4 loss components to get total mini-batch loss
        mb_loss = 0
        for k in loss_components.keys():
            mb_loss += loss_components[k]
            
        # backpropagate and update model parameters
        optimiser.zero_grad()
        mb_loss.backward()
        optimiser.step()
        
        # accumulate loss
        #mb_group_loss += mb_loss.item()
        epoch_loss += mb_loss.item()
        
        # print something periodically so we can monitor progress
        #if (bidx+1) % mb_group_size == 0:
        #    avg_loss_per_mb_over_group = mb_group_loss / mb_group_size
        #    #print(f"batch {bidx+1:4d}; avg loss per mb: {avg_loss_per_mb_over_group:.4f}")
        #    mb_group_loss = 0

        #if (bidx+1) >= 6:
        #   break

    # compute average loss per minibatch over the current epoch
    avg_loss_per_mb = epoch_loss / len(dataloader)

    print(f"epoch {epoch:3d}; avg loss per mb: {avg_loss_per_mb:.4f}")

    # manage the periodic saving of checkpoint files
    if epoch % n_epochs_checkpoint == 0:
        end_time = time.time()
        train_time = (end_time - start_time_checkpoint) / 60
        save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb)
        print(f"Checkpoint training time: {train_time:.2f} minutes\n")
        start_time_checkpoint = time.time()


# print total training time (in minutes)
end_time = time.time()
train_time = (end_time - start_time) / 60
print(f"\nTotal training time: {train_time:.2f} minutes\n")

# save a checkpoint file if training epochs have been processed since
# the last time a checkpoint file was saved
if epoch % n_epochs_checkpoint != 0:
    save_model_checkpoint(epoch, model, optimiser, avg_loss_per_mb)





















