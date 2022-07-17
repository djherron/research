#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:48:55 2022

@author: dave
"""

'''
This script implements algorithms for recall@k-based predictive
performance evaluation metrics for use with the VRD dataset. It
implements a Python version of the Lu & Fei-Fei (2016) recall@50
and recall@100 metrics that were originally implemented in MATLAB.
It also implements an adaptation of these metrics that we 
designed to be more sensitive to changes in the rankings of 
predicted VRs that we call 'Mean Avg Recall@k'.

VR stands for 'visual relationship'

Notes:
    
1) The predicted VRs for the Test Set images will have been stored on disk 
   in a json format identical to the format used for all VRD ground-truth 
   annotated VRs. There will, however, be one important exception to this: 
   the format of the bounding boxes in the predicted VRs will be the more
   conventional format of [xmin, ymin, xmax, ymax] rather than the format
   used by the VRD's annotated VRs, [ymin, ymax, xmin, xmax]. The reason
   for this is that the Faster RCNN object detector model outputs its
   bbox predictions in the format [xmin, ymin, xmax, ymax] and this format
   is maintained throughout the remainder of the visual relationship
   prediction pipeline.
   
2) The ground-truth (gt) annotated VRs for the Test Set images will be
   loaded from their host .json file.  The format of the bounding boxes
   in the gt VRs, [ymin, ymax, xmin, xmax], will be modified to the
   format [xmin, ymin, xmax, ymax] after the gt VRs have been loaded in
   order to have the same format as the bounding boxes in the predicted
   VRs.
'''


#%%

import json
import numpy as np
import pandas as pd

#%%

# set the top value of k for recall@k
topk = 100;

#%% load the Lu & Fei-Fei (2016) Test Set data exported from MATLAB

# load the groundtruth (gt) VR annotations of the Test Set images
filename = 'feifei_testset_relationships_groundtruth_stage_2.json'
with open(filename, 'r') as file:
    gt_relationships = json.load(file)

# load the groundtruth (gt) VR annotations of the Test Set images
filename = 'feifei_testset_relationships_predicted_stage_2.json'
with open(filename, 'r') as file:
    predicted_relationships = json.load(file)

#%% check number of image entries in each dictionary

print(len(gt_relationships))

print(len(predicted_relationships))


#%% define custom lists of image names to process

#img_idxs = [2,6,37,45,58,78,349,386,391,462,463,508,513,531];
image_names_1 = ['img_2', 'img_6', 'img_37', 'img_45', 'img_58', 'img_78',
                 'img_349', 'img_386', 'img_391', 'img_462', 'img_463',
                 'img_508', 'img_513', 'img_531']

#img_idxs2 = [551, 552, 553, 554, 555, 556, 557, 558, 559, 600];
image_names_2 = ['img_551', 'img_552', 'img_553', 'img_554', 'img_555', 
                 'img_556', 'img_557', 'img_558', 'img_559', 'img_600']

#img_idxs3 = [601,602,603,604,605,606,608,610,612,613,614,615,616,618];
image_names_3 = ['img_601', 'img_602', 'img_603', 'img_604', 'img_605', 
                 'img_606', 'img_608', 'img_610', 'img_612', 'img_613',
                 'img_614', 'img_615', 'img_616', 'img_618']

#%% set the list of image names to be processed

#image_subset_to_process = image_names_1
#image_subset_to_process = image_names_1 + image_names_2
image_subset_to_process = image_names_1 + image_names_2 + image_names_3


#%% Discussion of the 'predicted VR global DataFrame' (dfPG)

# The 'predicted VR global DF' (dfPG) holds all of the predicted VRs for
# all of the Test Set images that have predicted VRs. Each row of the dfPG
# holds all of the data associated with one predicted VR. The 'imname'
# column of the dfPG stores the name of the image with which each 
# predicted VR is associated.

# dfPG stands for: data frame Predicted Global

# A predicted VR consists of:
# - a 3-tuple, (s, p, o), of integer labels representing the classes of
#   the subject (s), predicate (p) and object (o) participating in the VR
# - a 4-tuple of integers representing the bbox of the 'subject' object
# - a 4-tuple of integers representing the bbox of the 'object' object
# - a real-valued confidence score of the prediction
# The dfPG has a column for each of these individual elements that are part
# of a predicted VR.

# The dfPG contains additional columns that are supplementary to  
# describing predicted VRs which are needed by the algorithms that 
# implement the computation of the predictive performance evaluation metrics.
# These supplementary columns are:
# - 'imname' (the filename of the image with which a predicted VR is
#   associated)
# - 'hit' (a binary column where a 1 indicates that the predicted VR has 
#   been deemed to be a hit, ie to match with a unique ground-truth,
#   annotated VR); a 'hit' is akin to a 'True Positive'
# - 'hitcs' (the cumulative sum of hits in the 'hit' column, over some 
#   range or set of rows in the dfPG)

# Notes re the Lu & Fei-Fei MATLAB recall@50 implementation:
# - the 'hit' column plays the role both of 1) the MATLAB 'tp' row vector
#   for each image and 2) the MATLAB tp_cell cell array for keeping track
#   of all the image-level 'tp' row vectors
# - the 'hit' and 'score' columns play the role of the MATLAB 'tp_all' 
#   and 'confs' variables used after the MATLAB main processing loop to 
#   stack all the image-level tp and confidences into global column vectors

#%% Build the dfPG

# Initialise the dfPG
# * establish the initial columns of the dfPG and add one row of data
#   representing a dummy predicted VR as a tactic to set the dtype
#   desired for each column 
# * the dummy predicted VR is removed once the dfPG build is complete

dfPG = pd.DataFrame({'imname': ['im0'],   # image name
                     'slb': [0],          # 'subject' object class label
                     'plb': [0],          # 'predicate' class label
                     'olb': [0],          # 'object' object class label
                     'sx1': [0],          # subject bbox xmin
                     'sy1': [0],          # subject bbox ymin
                     'sx2': [0],          # subject bbox xmax
                     'sy2': [0],          # subject bbox ymax
                     'ox1': [0],          #  object bbox xmin
                     'oy1': [0],          #  object bbox ymin
                     'ox2': [0],          #  object bbox xmax
                     'oy2': [0],          #  object bbox ymax
                     'score': [0.0]})     # prediction confidence score

# Iterate through the images and their associated predicted VRs.
# For images having one or more predicted VRs, transfer all of the data
# into the dfPG.
# Images with zero predicted VRs (a case expected to be rare but for
# which we nonetheless cater) are not represented in the dfPG.

# for testing
#image_names = image_subset_to_process

# for production
image_names = predicted_relationships.keys()

for imname in image_names:

    # initialise lists to hold the predicted VR data for an image
    imnames = []
    slabels = []
    plabels = []
    olabels = []
    sx1s = []
    sy1s = []
    sx2s = []
    sy2s = []
    ox1s = []
    oy1s = []
    ox2s = []
    oy2s = []
    scores = []    

    # get the predicted VRs for the current image
    imanno = predicted_relationships[imname]

    # transfer the predicted VR data for the current image to the lists    
    for vr in imanno:
        imnames.append(imname)
        slabels.append(vr['subject']['category'])
        plabels.append(vr['predicate'])
        olabels.append(vr['object']['category'])
        sx1s.append(vr['subject']['bbox'][0])
        sy1s.append(vr['subject']['bbox'][1])
        sx2s.append(vr['subject']['bbox'][2])
        sy2s.append(vr['subject']['bbox'][3])
        ox1s.append(vr['object']['bbox'][0])
        oy1s.append(vr['object']['bbox'][1])
        ox2s.append(vr['object']['bbox'][2])
        oy2s.append(vr['object']['bbox'][3])
        scores.append(vr['confidence'])

    # if the current image has zero predicted VRs, ignore it    
    if len(imnames) == 0:
        continue

    # put the predicted VR data for the current image into a dataframe
    dfPI = pd.DataFrame({'imname': imnames,   
                         'slb': slabels,       
                         'plb': plabels,       
                         'olb': olabels,       
                         'sx1': sx1s,          
                         'sy1': sy1s,          
                         'sx2': sx2s,          
                         'sy2': sy2s,          
                         'ox1': ox1s,          
                         'oy1': oy1s,          
                         'ox2': ox2s,          
                         'oy2': oy2s,          
                         'score': scores})     
 
    # sort the predicted VRs for the current image by the confidence
    # scores of the predictions, in descending order
    dfPI = dfPI.sort_values(by='score', ascending=False)

    # keep (up to) the 'top k' predicted VRs for the current image 
    # (as determined by the sorted prediction confidence scores) and 
    # discard the rest; many images will have fewer than 'k' predicted VRs,
    # in which case all of them will be retained
    dfPI = dfPI.iloc[0:topk]

    # concatenate the predicted VR dataframe for the current image to
    # the predicted VR global dataframe
    dfPG = dfPG.append(dfPI, ignore_index=True)

# remove the first row (the dummy predicted VR) from the dfPG
dfPG = dfPG.drop(labels=0, axis=0)

# add the column 'hit' initialised to all (integer) zeros that will be used
# to record which predicted VRs were found to be 'hits' (True Positives)
# that successfully predicted a particular gt VR
dfPG['hit'] = 0

# add the column 'iou' initialised to all (real) zeros that will be used 
# to record the IoU that the bboxes of a predicted VR marked as a 'hit'
# was found to have achieved with the bboxes of the gt VR with which it
# was matched (ie which it was deemed to have hit)
dfPG['iou'] = 0.0

#%% Examine the constructed dfPG

print(dfPG.shape)

#%%

print(dfPG.head())

#%%

mask = dfPG['imname'] == 'img_7'

print(dfPG[mask])

#%%

print(dfPG.tail(15))

#%%

image_names_dfPG = dfPG['imname'].unique()
n_images_dfPG = len(image_names_dfPG)
print(f'number of images in dfPG: {n_images_dfPG}')

#%% Discussion of the 'ground-truth VR global DataFrame' (dfGG)

# The 'ground-truth (gt) VR global DataFrame' (dfGG) holds all of the 
# gt VRs for all of the Test Set images that have gt VRs. Each row of the 
# dfGG holds all of the data associated with one gt VR. The 'imname'
# column of the dfGG stores the name of the image with which each 
# gt VR is associated.

# dfGG stands for: data frame Ground-truth Global

# A gt VR consists of:
# - a 3-tuple, (s, p, o), of integer labels representing the classes of
#   the subject (s), predicate (p) and object (o) participating in the VR
# - a 4-tuple of integers representing the bbox of the 'subject' object
# - a 4-tuple of integers representing the bbox of the 'object' object
# The dfGG has a column for each of these individual elements that are part
# of a gt VR.

# The dfGG contains additional columns that are supplementary to  
# describing gt VRs which are needed by the algorithms that implement
# the computation of the predictive performance evaluation metrics.
# These supplementary columns are:
# - 'imname' (the filename of the image with which a gt VR is associated)
# - 'hit' (a binary column where a 1 indicates that the gt VR has been hit
#   by a predicted VR; ie the gt VR has been matched with and, hence, deemed 
#   to have been successfully predicted by a unique predicted VR)

#%% Build the dfGG

# Indicate the bbox format used in the gt VR data being processed:
# True - indicates that the bbox format in the gt VR data loaded from disk
#        is [ymin, ymax, xmin, xmax]
# False - indicates that the bbox format in the gt VR data loaded from disk
#         is [xmin, ymin, xmax, ymax]
vrd_bbox_format = False

# Initialise the dfGG
# * establish the initial columns of the dfGG and add one row of data
#   representing a dummy gt VR as a tactic to set the dtype desired
#   for each column 
# * the dummy gt VR is removed once the dfPG build is complete

dfGG = pd.DataFrame({'imname': ['im0'],   # image name
                     'slb': [0],          # 'subject' object class label
                     'plb': [0],          # 'predicate' class label
                     'olb': [0],          # 'object' object class label
                     'sx1': [0],          # subject bbox xmin
                     'sy1': [0],          # subject bbox ymin
                     'sx2': [0],          # subject bbox xmax
                     'sy2': [0],          # subject bbox ymax
                     'ox1': [0],          #  object bbox xmin
                     'oy1': [0],          #  object bbox ymin
                     'ox2': [0],          #  object bbox xmax
                     'oy2': [0]})         #  object bbox ymax

# Iterate through the images and their associated gt VRs.
# For images having one or more gt VRs, transfer all of the data
# into the dfGG.
# Images with zero gt VRs (a case which does arise with the original
# annotations of the VRD dataset, but not with the customised annotations) 
# are not represented in the dfPG.

# for testing
#image_names = image_subset_to_process

# for production
image_names = gt_relationships.keys()

for imname in image_names:

    # initialise lists to hold the predicted VR data for an image
    imnames = []
    slabels = []
    plabels = []
    olabels = []
    sx1s = []
    sy1s = []
    sx2s = []
    sy2s = []
    ox1s = []
    oy1s = []
    ox2s = []
    oy2s = [] 

    # get the predicted VRs for the current image
    imanno = gt_relationships[imname]

    # transfer the gt VR data for the current image to the lists    
    for vr in imanno:
        imnames.append(imname)
        slabels.append(vr['subject']['category'])
        plabels.append(vr['predicate'])
        olabels.append(vr['object']['category'])
        if vrd_bbox_format:
            sx1s.append(vr['subject']['bbox'][2])
            sy1s.append(vr['subject']['bbox'][0])
            sx2s.append(vr['subject']['bbox'][3])
            sy2s.append(vr['subject']['bbox'][1])
            ox1s.append(vr['object']['bbox'][2])
            oy1s.append(vr['object']['bbox'][0])
            ox2s.append(vr['object']['bbox'][3])
            oy2s.append(vr['object']['bbox'][1])            
        else:
            sx1s.append(vr['subject']['bbox'][0])
            sy1s.append(vr['subject']['bbox'][1])
            sx2s.append(vr['subject']['bbox'][2])
            sy2s.append(vr['subject']['bbox'][3])
            ox1s.append(vr['object']['bbox'][0])
            oy1s.append(vr['object']['bbox'][1])
            ox2s.append(vr['object']['bbox'][2])
            oy2s.append(vr['object']['bbox'][3])

    # if the current image has zero gt VRs, ignore it    
    if len(imnames) == 0:
        continue

    # build a dataframe to hold the gt VR data for the current
    # image temporarily    
    dfGI = pd.DataFrame({'imname': imnames,   
                         'slb': slabels,       
                         'plb': plabels,       
                         'olb': olabels,       
                         'sx1': sx1s,          
                         'sy1': sy1s,          
                         'sx2': sx2s,          
                         'sy2': sy2s,          
                         'ox1': ox1s,          
                         'oy1': oy1s,          
                         'ox2': ox2s,          
                         'oy2': oy2s})    

    # add the gt VR data for the current image to the dfGG
    dfGG = dfGG.append(dfGI, ignore_index=True)

# remove the first row (the dummy gt VR) from the dfGG
dfGG = dfGG.drop(labels=0, axis=0)

# add the column 'hit' initialised to all zeros
dfGG['hit'] = 0

#%% Examine the constructed dfGG

print(dfGG.shape)

#%%

print(dfGG.head())

#%%

mask = dfGG['imname'] == 'img_2'

print(dfGG[mask])

#%%

print(dfGG.tail(15))

#%%

image_names_dfGG = dfGG['imname'].unique()
n_images_dfGG = len(image_names_dfGG)
print(f'number of images in dfGG: {n_images_dfGG}')

#%%

# set the bbox IoU minimum threshold to be exceeded for a match between
# pairs of predicted and gt bboxes to be declared to have occurred
iou_thresh = 0.5

# get the unique set of image names for which we have predicted VRs in the dfPG
image_names_dfPG = dfPG['imname'].unique()

#%%  Notes wrt the Fei-Fei data exported from MATLAB

# Image entries within the MATLAB data that had specific conditions:
#
# images with 1 hit:
# img_6, img_58, img_349, img_508, img_531
#
# images with 2 hits:
# img_608, img_618
#
# images with 3 hits:
# img_603, img_604, img_614, img_616 
#
# images with 4 hits:
# img_552, img_553, img_554, img_600
#
# images with 1 gt VR: 41
# img seq num: 73, 108, 138, 298
#  nr of hits:  0,   0,   1,   1
#
# images with 2 gt VRs: 56
# img seq num: 19, 140, 282, 285
#  nr of hits:  0,   0,   1,   1
#
# images with 1 pred VR: 0
# ...none...
# images with 2 pred VRs: 36
# img seq num: 77, 156, 268
#  nr of hits:  0,   0,   2


# Images found to be especially good for testing, ie that have lots of 
# partial/potential hits that exercise well the various conditions of the
# inner-most 'for' loop over the gt VRs
#
# img_600, img_603, ...


#%% (optional) override the list of image names to be processed

#image_names_dfPG = ['img_268']

#%% main processing loop

progress_freq = 50
cnt = 0    

for imname in image_names_dfPG:

    cnt += 1
    if cnt % progress_freq == 0:
        print(f'imname: {imname}')
    #print(f'imname: {imname}')
    
    # get the predicted VRs for the current image
    dfPG_mask = dfPG['imname'] == imname
    dfPI = dfPG[dfPG_mask]
    #print('dfPI')
    #print(dfPI)
    n_pred_vrs = dfPI.shape[0]
    #print(f'n_pred_vrs: {n_pred_vrs}')
    pred_hits = [0] * n_pred_vrs
    pred_iou = [0.0] * n_pred_vrs
    
    # get the gt VRs for the current image
    dfGG_mask = dfGG['imname'] == imname
    if np.sum(dfGG_mask) == 0:
        raise ValueError(f"dfPG image '{imname}' has no match in dfGG")
    dfGI = dfGG[dfGG_mask]
    #print()
    #print('dfGI')
    #print(dfGI)
    n_gt_vrs = dfGI.shape[0]
    #print(f'n_gt_vrs: {n_gt_vrs}')
    gt_hits = [0] * n_gt_vrs
    
    # iterate over the predicted VRs for the current image
    for idxp in range(n_pred_vrs):
        #print()
        #print(f'pred VR idxp {idxp}')
        
        # get the (subject, predicate, object) class labels for the
        # current predicted VR
        p_slabel = dfPI.iloc[idxp]['slb']
        p_plabel = dfPI.iloc[idxp]['plb']
        p_olabel = dfPI.iloc[idxp]['olb']
        
        # get the subject bbox for the current predicted VR
        p_sx1 = dfPI.iloc[idxp]['sx1']
        p_sy1 = dfPI.iloc[idxp]['sy1']      
        p_sx2 = dfPI.iloc[idxp]['sx2']        
        p_sy2 = dfPI.iloc[idxp]['sy2']      
        
        # get the object bbox for the current predicted VR
        p_ox1 = dfPI.iloc[idxp]['ox1']
        p_oy1 = dfPI.iloc[idxp]['oy1']      
        p_ox2 = dfPI.iloc[idxp]['ox2']        
        p_oy2 = dfPI.iloc[idxp]['oy2']        

        # initialise our variable for tracking the index of the gt VR 
        # found to have the best overall bbox IoU (Intersection-over-Union)
        # with the current predicted VR
        best_idx = -1
        
        # initialise our variable for tracking the value of the best bbox IoU
        best_iou = -1
             
        # iterate over the gt VRs for the current image and look for the
        # 'best hit' with respect to the current predicted VR
        for idxg in range(n_gt_vrs):
            #print(f'idxg {idxg}')
            
            # if the (s, p, o) class labels of the current predicted VR
            # don't match those of the current gt VR, the current gt VR
            # is not a candidate hit; so we're done with it
            if p_slabel == dfGI.iloc[idxg]['slb'] and \
               p_plabel == dfGI.iloc[idxg]['plb'] and \
               p_olabel == dfGI.iloc[idxg]['olb']:
                pass
            else:
                continue

            #print(f'idxg {idxg} is an (s,p,o) label match')
     
            # if the current gt VR is already a 'hit' with respect to some
            # other predicted VR for the current image, we're done with it;
            # (a gt VR can't be 'hit' by more than one predicted VR)
            #if dfGI.iloc[idxg]['hit'] > 0:
            #if dfGI.loc[idxg, 'hit'] > 0:
            if gt_hits[idxg] > 0:
                continue
            
            #print(f'idxg {idxg} is not already a hit')
            
            # calculate the width & height of the intersection of
            # the subject bboxes of the two VRs
            s_max_x1 = np.max([p_sx1, dfGI.iloc[idxg]['sx1']])
            s_max_y1 = np.max([p_sy1, dfGI.iloc[idxg]['sy1']])            
            s_min_x2 = np.min([p_sx2, dfGI.iloc[idxg]['sx2']])
            s_min_y2 = np.min([p_sy2, dfGI.iloc[idxg]['sy2']])
            s_intersect_width = s_min_x2 - s_max_x1 + 1
            s_intersect_height = s_min_y2 - s_max_y1 + 1

            # calculate the width & height of the intersection of
            # the object bboxes of the two VRs
            o_max_x1 = np.max([p_ox1, dfGI.iloc[idxg]['ox1']])
            o_max_y1 = np.max([p_oy1, dfGI.iloc[idxg]['oy1']])            
            o_min_x2 = np.min([p_ox2, dfGI.iloc[idxg]['ox2']])
            o_min_y2 = np.min([p_oy2, dfGI.iloc[idxg]['oy2']])
            o_intersect_width = o_min_x2 - o_max_x1 + 1
            o_intersect_height = o_min_y2 - o_max_y1 + 1

            # if the subject bboxes of the two VRs intersect, and the object 
            # bboxes of the two VRs also intersect, then we potentially
            # have a hit; otherwise, we're done with the current gt VR
            if s_intersect_width > 0 and s_intersect_height > 0 and \
               o_intersect_width > 0 and o_intersect_height > 0:
                pass
            else:
                continue
            
            #print(f'idxg {idxg} has positive sub/obj bbox intersection')
            
            # calculate the intersection area of the subject bbox pair
            s_intersect_area = s_intersect_width * s_intersect_height
            
            # calculate the union area of the subject bbox pair
            p_s_width = p_sx2 - p_sx1 + 1
            p_s_height = p_sy2 - p_sy1 + 1
            p_s_area = p_s_width * p_s_height
            g_s_width = dfGI.iloc[idxg]['sx2'] - dfGI.iloc[idxg]['sx1'] + 1
            g_s_height = dfGI.iloc[idxg]['sy2'] - dfGI.iloc[idxg]['sy1'] + 1
            g_s_area = g_s_width * g_s_height
            s_union_area = p_s_area + g_s_area - s_intersect_area
            
            # calculate the IoU of the subject bbox pair
            s_iou = s_intersect_area / s_union_area
 
            # calculate the intersection area of the object bbox pair
            o_intersect_area = o_intersect_width * o_intersect_height           
 
            # calculate the union area of object bbox pair
            p_o_width = p_ox2 - p_ox1 + 1
            p_o_height = p_oy2 - p_oy1 + 1
            p_o_area = p_o_width * p_o_height
            g_o_width = dfGI.iloc[idxg]['ox2'] - dfGI.iloc[idxg]['ox1'] + 1
            g_o_height = dfGI.iloc[idxg]['oy2'] - dfGI.iloc[idxg]['oy1'] + 1
            g_o_area = g_o_width * g_o_height
            o_union_area = p_o_area + g_o_area - o_intersect_area            
            
            # calculate the IoU of the object bbox pair                
            o_iou = o_intersect_area / o_union_area
            
            # get the smaller of the subject bbox and object bbox IoUs
            iou = np.min([s_iou, o_iou])
            
            #print(f'idxg {idxg} has IoU {iou}')

            # if the smaller of the two IoUs exceeds the IoU threshold,
            # then both of them do; so the current gt VR is a 'candidate hit'
            if iou > iou_thresh:
                #print(f'idxg {idxg} IoU {iou} > iou_thresh {iou_thresh}')
                # if the smaller of the two IoUs exceeds the best IoU found
                # so far (amongst the gt VRs for the current image), then
                # the current gt VR is the 'best candidate hit' found so far,
                # so keep track of it
                if iou > best_iou: 
                    #print(f'idxg {idxg} IoU {iou} > best_iou {best_iou}')
                    best_iou = iou 
                    best_idx = idxg
                    
        # if we found a gt VR 'best hit' for the current predicted VR, then
        # record this fact; mark the predicted VR as 'being' a 'hit' (a True
        # Positive), and mark the gt VR as 'having been' 'hit' (so it can't
        # get hit again)             
        if best_idx >= 0:
            pred_hits[idxp] = 1
            pred_iou[idxp] = np.round(best_iou,4)
            gt_hits[best_idx] = 1
        #    print(f'pred VR idxp {idxp} is a hit with gt VR idxg {best_idx}')
        #else:
        #    print(f'pred VR idxp {idxp} is NOT a hit') 

    # transfer the information about which predicted VRs for the current
    # image are 'hits' (True Positives) back to the predicted VR global 
    # dataframe, dfPG; (this information will be further processed later)
    dfPG.loc[dfPG_mask, 'hit'] = pred_hits
    
    # transfer the information about the bbox IoU achieved between the
    # bboxes of the predicted VRs that were 'hits' and the bboxes of 
    # the gt VRs they hit (ie with which they were matched); (this 
    # information may be of interest during qualitative predictive
    # performance evaluation)
    dfPG.loc[dfPG_mask, 'iou'] = pred_iou
        
    # transfer the information about which gt VRs for the current image
    # were 'hit' back to the gt VR global dataframe, dfGG; (this 
    # information isn't required for computing measures of our metrics,
    # but it will have great value in terms of facilitating qualitative
    # predictive performance evaluation) 
    dfGG.loc[dfGG_mask, 'hit'] = gt_hits


#%% Examine the dfPG and dfGG after the main processing loop


#%%

print(dfPG.head())

#%%

mask = dfPG['imname'] == 'img_268'
print(dfPG[mask])

#%%

print(dfPG['hit'].sum())

print(dfPG.shape)

#%%

print(dfGG.head())

#%%

mask = dfGG['imname'] == 'img_268'
print(dfGG[mask])

#%%

print(dfGG['hit'].sum())

print(dfGG.shape)



#%% The hard part is done. Now we can compute measures for our metrics.
     
   
#%% Metric: Global Recall@N

# This is the metric used by Lu & Fei-Fei (2016), and by Donadello (2019).
# It does not calculate recall@N for each image and then take the mean of
# those image-level measures. Instead, it considers all of the predicted VRs
# in relation to all of the gt VRs, over the entire Test Set. Thus, we
# describe this metric as being a 'global' metric rather than an 'image-level'
# metric and therefore refer to it as 'Global Recall@N', where N is either
# 50 or 100.

# sort all of the predicted VRs in the global dataframe (dfPG) by their 
# prediction confidence scores
# (create a copy of the dfPG so that we don't disturb the original)
dfPG2 = dfPG.sort_values(by='score', ascending=False)

# compute the cumulative sum of the hits marked in the 'hit' column
# and store the result in a new column of the dataframe
dfPG2['hitcs'] = dfPG2['hit'].cumsum()

# get the total number of gt VRs
total_num_gt_vrs = dfGG.shape[0]

# compute recall@k for every value of k across the entire dataframe
dfPG2['recall'] = dfPG2['hitcs'] / total_num_gt_vrs

# get the value of recall in the bottom row of the dataframe
idx = dfPG2.shape[0]
recallN = dfPG2['recall'].iloc[idx-1]
global_recall_at_N = np.round(100 * recallN, 5)

print(f'Global Recall@N (%): {global_recall_at_N}')


#%% Metric: Mean Recall@N Per Image

# Here we calculate recall@N for each image individually. Then we
# take the mean of the image-level measures of recall@N to give
# 'mean recall@N per image'.  We use N = 50 or 100.

# NOTE: For this metric we process the images based on the image names
# present in the dfGG dataframe (in which all images having a non-zero
# number of gt VRs are present). It is possible that an image with
# non-zero gt VRs may (for some reason) have had zero predicted VRs
# (eg if the object detection failed to identify at least 2 objects or,
# potentially, for other reasons as well). But images with zero predicted 
# VRs aren't present in the dfPG dataframe because we've chosen not to 
# support such scenarios (eg by allowing an image to have one dummy row 
# that represents a 'missing' VR). Thus, if we drove the processing for 
# this metric based on the image names present in the dfPG we could 
# potentially inadvertently miss some images that should participate in
# the calculation of the metric. The measure of recall for these images is
# zero (because they have zero predicted VRs but non-zero gt VRs). And 
# these zero-valued measures of recall should participate in the calculation
# of the 'mean' recall per image.  If these zero-valued measures of recall
# are excluded from the calculation of 'mean recall per image', the 
# resulting measure of 'mean recall per image' will be mistakenly
# inflated (ie higher than the correct figure).  By driving the processing
# of this metric based on the image names present in the dfGG, we avoid
# inadvertently excluding these zero-valued measures of recall from the
# calculation of the metric and ensure we compute the correct value for
# the metric.

# (NOTE: It is also possible that an image has zero gt VRs, but in all
# such cases encountered so far, these images also had zero predicted VRs.
# The Lu & Fei-Fei (2016) original VRD annotations had many such cases.  
# In such cases, the image has no presence in either the dfPG or the dfGG
# and therefore does not participate in predictive performance evaluation
# in any way. With respect to predictive performance evaluation, such 
# images are, effectively, not part of the dataset.)

image_names_dfGG = dfGG['imname'].unique()

recall_per_image = []

for imname in image_names_dfGG:

    dfPG_mask = dfPG['imname'] == imname 
    n_hits = dfPG['hit'][dfPG_mask].sum()

    dfGG_mask = dfGG['imname'] == imname
    n_gt_vrs = dfGG[dfGG_mask].shape[0]
    
    recall = n_hits / n_gt_vrs
    
    recall_per_image.append(recall)
    
    
mean_recall = np.round(100 * np.mean(recall_per_image), 5)

print(f'Mean Recall@N per image (%): {mean_recall}')


#%% Metric: Mean Average Recall@k Top-N

# We use N = 50 or 100.

image_names_dfGG = dfGG['imname'].unique()

avg_recall_at_k_per_image = []

for imname in image_names_dfGG:

    dfPG_mask = dfPG['imname'] == imname
    n_pred_vrs = dfPG[dfPG_mask].shape[0]
    
    if n_pred_vrs > 0:
        hit_cumsum = dfPG['hit'][dfPG_mask].cumsum()

        dfGG_mask = dfGG['imname'] == imname
        n_gt_vrs = dfGG[dfGG_mask].shape[0]
    
        recall_at_k = hit_cumsum / n_gt_vrs

        # get the subrange of values of recall_at_k over which
        # we wish to take the average 
        if n_gt_vrs <= n_pred_vrs:
            idx1 = n_gt_vrs - 1
            idx2 = n_pred_vrs
        else:
            idx1 = n_pred_vrs - 1
            idx2 = n_pred_vrs    
        recall_at_k_subrange = recall_at_k[idx1:idx2]
    
        avg_recall_at_k = np.mean(recall_at_k_subrange)
    else:
        avg_recall_at_k = 0.0
    
    avg_recall_at_k_per_image.append(np.round(avg_recall_at_k, 5))

mean_avg_recall_at_k = np.round(100 * np.mean(avg_recall_at_k_per_image), 5)

print(f'Mean Avg Recall@k Top-N (%): {mean_avg_recall_at_k}')


#%% save dfPG dataframe to csv file

filename = 'feifei_dfpg_top100.csv'
dfPG.to_csv(filename, index=False)

#%% save dfPG dataframe to csv file

filename = 'feifei_dfgg_top100.csv'
dfGG.to_csv(filename, index=False)


#%% Records of results with my metrics

#%% Lu & Fei-Fei (2016) Full Test Set Results

# Total number of hits:
# Top-50 : 1078
# Top-100: 1143

# Globall Recall@50 (%) : 14.11364
# Globall Recall@100 (%): 14.96465

# Mean Recall@50 per image (%) : 14.05819
# Mean Recall@100 per image (%): 14.63294

# Mean Avg Recall@k Top-50 (%) : 12.38397
# Mean Avg Recall@k Top-100 (%): 13.11312








