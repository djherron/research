#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 18:50:21 2022

@author: dave
"""

'''
Utility functions relating to bounding boxes and pairs of bounding boxes.
These utility functions derive (extract) geometric features of individual 
bounding boxes and of ordered pairs of bounding boxes.

These functions are intended to be used primarily to process pairs of 
predicted bboxes output by a Faster RCNN object detector model and 
pre-arranged into (subject, object) ordered pairs for input into a 2nd
neural network whose task will be to predict the predicate(s) that best
describe visual relationships between the ordered pairs of objects 
predicted to reside within the bounding boxes.

The utility functions defined here will be used to pre-process
the ordered pairs of bboxes in order to derive geometric features
describing geometric relations between the bboxes.
These derived geometric features will act as additional input data to 
the 2nd neural network to supplement the bbox specifications and object
class predictions of each ordered pair of objects. 

NOTE: All functions here assume that the bbox coordinates are in the format
[xmin, ymin, xmax, ymax]. This is the format of the bounding boxes output
by a Faster RCNN object detector model. It is NOT the format used in the
visual relationship annotations for images of the VRD dataset, which is
[ymin, ymax, xmin, xmax]. 

TODO: Consider revising the calculations of widths and heights, etc, to use
width = b[2] - b[0] + 1
height = b[3] - b[1] + 1
as was done in an online example we found, and as is done in the MATLAB code
for the Lu & Fei-Fei (2016) paper. Is this a safety tactic, to avoid values
of zero (0)?  It's not at all clear that this is the 'mathematically
correct' way to do it.  Further, recall that in City
module INM705 (Image Analysis), Alex Ter-Sarkisov did NOT add the 1. And
I have found other examples online
(eg https://medium.com/analytics-vidhya/basics-of-bounding-boxes-94e583b5e16c)
where they do NOT add the 1 to calculate the width and height. 
'''

#%%

import math

#%% bbox width and height

def bb_width_height(b):
    width = float(b[2] - b[0])
    height = float(b[3] - b[1])
    return width, height

#%% bbox areas

def bb_areas(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_area = b1_width * b1_height
    b2_width, b2_height = bb_width_height(b2)
    b2_area = b2_width * b2_height
    return b1_area, b2_area

#%% bbox centroids

def bb_centroids(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_c_x = b1[0] + (b1_width / 2)
    b1_c_y = b1[1] + (b1_height / 2)
    b2_width, b2_height = bb_width_height(b2)
    b2_c_x = b2[0] + (b2_width / 2)
    b2_c_y = b2[1] + (b2_height / 2)   
    return (b1_c_x, b1_c_y), (b2_c_x, b2_c_y)

#%% Euclidean distance between bbox centroids

def bb_euclidean_distance(b1, b2):
    b1_c, b2_c = bb_centroids(b1,b2)
    eucl_dist = math.sqrt( (b1_c[0] - b2_c[0])**2 + (b1_c[1] - b2_c[1])**2 )
    return eucl_dist

#%% sine and cosine of angle between bbox centroids

def bb_sine_and_cosine_of_angle_between_centroids(b1, b2):
    '''
    Treat the centroid of bbox b1 as though it were the origin. 
    Calculate the sine and cosine of the angle between the two centroids
    by moving counter-clockwise relative to the centroid of bbox b1.
    '''
    hypotenuse_length = bb_euclidean_distance(b1,b2)
    b1_c, b2_c = bb_centroids(b1,b2)
    vert_length = b2_c[1] - b1_c[1]
    sine_theta = vert_length / hypotenuse_length
    horiz_length = b2_c[0] - b1_c[0]
    cosine_theta = horiz_length / hypotenuse_length
    return sine_theta, cosine_theta

#%% bbox aspect ratios

def bb_aspect_ratios(b1, b2):
    b1_width, b1_height = bb_width_height(b1)
    b1_aspect_ratio = b1_height / b1_width
    b2_width, b2_height = bb_width_height(b2)
    b2_aspect_ratio = b2_height / b2_width
    return b1_aspect_ratio, b2_aspect_ratio

#%% bbox intersection area

def bb_intersection_area(b1, b2):
    x1 = max(b1[0], b2[0])  # max xmin
    y1 = max(b1[1], b2[1])  # max ymin
    x2 = min(b1[2], b2[2])  # min xmax
    y2 = min(b1[3], b2[3])  # min ymax
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    return float(intersection_area)

#%% bbox union area

def bb_union_area(b1, b2):
    intersection_area = bb_intersection_area(b1,b2)   
    b1_area, b2_area = bb_areas(b1,b2)
    union_area = b1_area + b2_area - intersection_area
    return union_area

#%% bbox IoU (Intersection over Union)

def bb_intersection_over_union(b1, b2):
    '''
    IoU is always in the unit interval [0,1].
    '''
    ia = bb_intersection_area(b1,b2)
    ua = bb_union_area(b1,b2)
    iou = ia / ua
    return iou

#%% bbox inclusion ratios

def bb_inclusion_ratios(b1, b2):
    '''
    Calculate the inclusion ratios for a pair of bounding boxes.
    An inclusion ratio measures the degree to which one bbox is 
    enclosed within another bbox. Inclusion ratios always lie in
    the unit interval [0,1]. If two bboxes are identical, both
    inclusion ratios will be exactly 1, which corresponds to an
    IoU of 1.
    '''
    b1_area, b2_area = bb_areas(b1,b2)
    intersection_area = bb_intersection_area(b1,b2)
    ir_b1b2 = intersection_area / b1_area
    ir_b2b1 = intersection_area / b2_area
    return ir_b1b2, ir_b2b1

#%% bbox area ratios

def bb_area_ratios(b1, b2):
    '''
    Calculate the area ratios for a pair of bounding boxes. An area ratio
    measures the ratio of the area of one bbox relative to the area of the
    other bbox. Area ratios always lie in the interval (0, infty).  
    '''
    b1_area, b2_area = bb_areas(b1,b2)  
    ar_b1b2 = b1_area / b2_area 
    ar_b2b1 = b2_area / b1_area 
    return ar_b1b2, ar_b2b1

#%% ratios of bbox area to image area

def bb_area_to_image_area_ratios(b1, b2, im_w, im_h):
    b1_area, b2_area = bb_areas(b1,b2)
    im_area = im_w * im_h
    b1_to_im_area_ratio = b1_area / im_area
    b2_to_im_area_ratio = b2_area / im_area
    return b1_to_im_area_ratio, b2_to_im_area_ratio

#%%

def bb_horiz_dist_edges(b1, b2, im_w):
    '''
    The horizontal distance between the right and left edges
    of two bboxes as a ratio relative to the image width.
    '''
    horiz_dist = 0
    if b1[2] < b2[0]:               # b1 is to the left of b2
        horiz_dist = b2[0] - b1[2]
    if b2[2] < b1[0]:               # b2 is to the left of b1
        horiz_dist = b1[0] - b2[2]
    horiz_dist_to_im_width_ratio = horiz_dist / im_w
    return horiz_dist_to_im_width_ratio

#%%

def bb_vert_dist_edges(b1, b2, im_h):
    '''
    The vertical distance between the bottom and top edges
    of two bboxes as a ratio relative to the image height.
    '''
    vert_dist = 0
    if b1[3] < b2[1]:               # b1 is below b2
        vert_dist = b2[1] - b1[3]
    if b2[3] < b1[1]:               # b2 is below b1
        vert_dist = b1[1] - b2[3]
    vert_dist_to_im_height_ratio = vert_dist / im_h
    return vert_dist_to_im_height_ratio

#%% 

def bb_horiz_dist_centroids(b1, b2, im_w):
    '''
    The horizontal distance between the centroids of two bboxes 
    as a ratio relative to the image width.
    '''
    b1_c, b2_c = bb_centroids(b1,b2)
    horiz_dist = abs(b1_c[0] - b2_c[0])
    horiz_dist_to_im_width_ratio = horiz_dist / im_w
    return horiz_dist_to_im_width_ratio

#%%

def bb_vert_dist_centroids(b1, b2, im_h):
    '''
    The vertical distance between the centroids of two bboxes 
    as a ratio relative to the image height.
    '''
    b1_c, b2_c = bb_centroids(b1,b2)
    vert_dist = abs(b1_c[1] - b2_c[1])
    vert_dist_to_im_height_ratio = vert_dist / im_h
    return vert_dist_to_im_height_ratio
















