#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 17:41:10 2022

@author: dave
"""

'''
This is a script for testing the utility functions in vrd_utils5.py
'''

#%%

import math
import vrd_utils5 as vrdu5

#%% bbox width and height

bb = [1,1,4,4]
bb_width, bb_height = vrdu5.bb_width_height(bb)
assert bb_width == 3.0
assert bb_height == 3.0

#%% bbox areas

b1 = [1,1,4,4]
b2 = [2,2,6,6]
b1_area, b2_area = vrdu5.bb_areas(b1,b2)
assert b1_area == 9.0
assert b2_area == 16.0

#%% bbox centroids

b1 = [1,1,5,5]
b2 = [2,2,6,6]
b1_c, b2_c = vrdu5.bb_centroids(b1,b2)
assert b1_c == (3.0, 3.0)
assert b2_c == (4.0, 4.0)

#%% Euclidean distance between bbox centroids

b1 = [1,1,5,5]
b2 = [2,2,6,6]
eucl_dist = vrdu5.bb_euclidean_distance(b1,b2)
assert eucl_dist == math.sqrt(2)

#%% sine and cosine of angle between bbox centroids

# b2 centroid is directly above b1 centroid
b1 = [5,5,9,9]     # centroid is (7,7)
b2 = [5,10,9,14]   # centroid is (7,12)
res = vrdu5.bb_sine_and_cosine_of_angle_between_centroids(b1,b2)
sine_theta, cosine_theta = res
assert sine_theta == 1.0
assert cosine_theta == 0.0

# b2 centroid is directly to the left of b1 centroid
b1 = [5,5,9,9]   # centroid is (7,7)
b2 = [0,5,4,9]   # centroid is (2,7)
res = vrdu5.bb_sine_and_cosine_of_angle_between_centroids(b1,b2)
sine_theta, cosine_theta = res
assert sine_theta == 0.0
assert cosine_theta == -1.0

# b2 centroid is directly below b1 centroid
b1 = [5,5,9,9]   # centroid is (7,7)
b2 = [5,0,9,4]   # centroid is (7,2)
res = vrdu5.bb_sine_and_cosine_of_angle_between_centroids(b1,b2)
sine_theta, cosine_theta = res
assert sine_theta == -1.0
assert cosine_theta == 0.0

# b2 centroid is directly to the right of b1 centroid
b1 = [5,5,9,9]    # centroid is (7,7)
b2 = [10,5,14,9]  # centroid is (12,7)
res = vrdu5.bb_sine_and_cosine_of_angle_between_centroids(b1,b2)
sine_theta, cosine_theta = res
assert sine_theta == 0.0
assert cosine_theta == 1.0

#%% bbox aspect ratios (height over width)

b1 = [1,1,5,3]
b2 = [2,2,4,6]
b1_ar, b2_ar = vrdu5.bb_aspect_ratios(b1,b2)
assert b1_ar == 0.5
assert b2_ar == 2.0

#%% bbox intersection area

#b1 = [1,1,4,4]
#b2 = [2,2,5,5]
b1 = [370, 379, 393, 396]
b2 = [369, 380, 397, 400]
ia = vrdu5.bb_intersection_area(b1,b2)
#assert ia == 4.0
print(ia)

#%% bbox union area

#b1 = [1,1,4,4]
#b2 = [2,2,5,5]
b1 = [370, 379, 393, 396]
b2 = [369, 380, 397, 400]
ua = vrdu5.bb_union_area(b1,b2)
#assert ua == 14.0
print(ua)

#%%  bbox IoU (Intersection over Union)

b1 = [1,1,4,4]
b2 = [2,2,5,5]
#b1 = [370, 379, 393, 396]
#b2 = [369, 380, 397, 400]
iou = vrdu5.bb_intersection_over_union(b1,b2)
assert iou == 4.0 / 14.0
#print(iou)

b1 = [1,1,5,5]
b2 = [1,1,5,5]
iou = vrdu5.bb_intersection_over_union(b1,b2)
assert iou == 1.0

#%% bbox inclusion ratios

b1 = [1,1,4,4]
b2 = [2,2,6,6]
ir_b1b2, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1,b2)
assert ir_b1b2 == 4.0 / 9.0
assert ir_b2b1 == 0.25

# two identical bboxes
b1 = [1,1,5,5]
b2 = [1,1,5,5]
ir_b1b2, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1,b2)
assert ir_b1b2 == 1.0
assert ir_b2b1 == 1.0

# two bboxes that don't intersect
b1 = [1,1,5,5]
b2 = [6,1,10,5]
ir_b1b2, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1,b2)
assert ir_b1b2 == 0.0
assert ir_b2b1 == 0.0

#%% bbox area ratios

b1 = [1,1,4,4]
b2 = [2,2,6,6]
ar_b1b2, ar_b2b1 = vrdu5.bb_area_ratios(b1,b2)
assert ar_b1b2 == 9.0 / 16.0  # 0.5625
assert ar_b2b1 == 16.0 / 9.0  # 1.777...7778

#%% bbox area to image area ratios

b1 = [1,1,481,481]   # 480 *  480 = 230400
b2 = [60,4,120,124]  # 60 * 120 = 7200
im_w = 800
im_h = 600
# im_area = 800 * 600 = 480000
# 230400 / 480000 = 0.48
# 7200 / 480000 = 0.015
ba2ia_b1, ba2ia_b2 = vrdu5.bb_area_to_image_area_ratios(b1,b2,im_w,im_h)
assert ba2ia_b1 == 0.48
assert ba2ia_b2 == 0.015
#print(ba2ia_b1, ba2ia_b2)

#%% horizontal distance between right and left edges of two bboxes
# (as a ratio relative to the image width)

b1 = [1,1,5,5]
b2 = [25,2,30,40]
im_w = 200
# horiz_dist = 25 - 5 - 20
# ratio - 20 / 200 = 0.1
hd2iw = vrdu5.bb_horiz_dist_edges(b1,b2,im_w)
assert hd2iw == 0.1

#%% vertical distance between bottom and top edges of two bboxes
# (as a ratio relative to the image height)

b1 = [50,50,80,80]
b2 = [1,1,10,10]
im_h = 200
# vert_dist = 50 - 10 = 40
# ratio = 40 / 200 = 0.2
vd2ih = vrdu5.bb_vert_dist_edges(b1,b2,im_h)
assert vd2ih == 0.2

#%% horizontal distance between bbox centroids
# (as a ratio relative to the image width)

b1 = [0,0,4,8]  # centroid (2,4)
b2 = [3,1,7,5]  # centroid (5,3)
im_w = 12
hdc2iw = vrdu5.bb_horiz_dist_centroids(b1,b2,im_w)
assert hdc2iw == 0.25

#%% vertical distance between bbox centroids
# (as a ratio relative to the image height)

b1 = [0,0,4,8]  # centroid (2,4)
b2 = [3,1,7,5]  # centroid (5,3)
im_h = 10
vdc2ih = vrdu5.bb_vert_dist_centroids(b1,b2,im_h)
assert vdc2ih == 0.1






































