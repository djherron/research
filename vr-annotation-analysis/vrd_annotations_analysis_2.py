#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 15:42:24 2021

@author: dave
"""

'''
This script finds subsets of VRD training images according to specialised
criteria in support of discovering which images have annotations that
require customisation as part of cleaning the VRD dataset.
'''

#%%

import os
import numpy as np
import vrd_utils as vrdu
import vrd_utils2 as vrdu2

#%% get the VRD data

# set path to directory in which the original VRD annotations reside
anno_dir = os.path.join('.', 'data', 'annotations_original')

# get the original VRD object class names
path = os.path.join(anno_dir, 'objects.json')
vrd_objects = vrdu.load_VRD_object_class_names(path)

# get the original VRD predicate names
path = os.path.join(anno_dir, 'predicates.json')
vrd_predicates = vrdu.load_VRD_predicate_names(path)

# get the original VRD visual relationship annotations
path = os.path.join(anno_dir, 'annotations_train.json')
vrd_anno = vrdu.load_VRD_image_annotations(path)

# get the original set of VRD training image names
vrd_img_names = list(vrd_anno.keys())


#%% analysis X - object class 'glasses'

#%% X.1

# get all images whose visual relationship annotations refer to the 
# object class 'glasses'
cls_name = 'glasses'
cls_idx = vrd_objects.index(cls_name)

images_with_target_cls = []
annos_with_target_cls = []
for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)    
    if cls_idx in all_classes:
        images_with_target_cls.append(imname)
        annos_with_target_cls.append(imanno)

print(f'number of images found: {len(images_with_target_cls)}')

#%% X.2 

# Of the images that refer to object class 'glasses', find the subset of
# images that are likely using 'glasses' to refer to 'drinking glasses'
# rather than to 'eyeglasses'

# set up visual relationship exclusions
vr_exclusions = [('person', 'wear', 'glasses'), ('glasses', 'on', 'person'),
                 ('person', 'has', 'glasses'), ('glasses', 'on', 'face'),
                 ('helmet', 'above', 'glasses'), ('person', 'in', 'glasses')]

images_to_review = []
annos_to_review = []
for idx, imname in enumerate(images_with_target_cls):
    imanno = vrd_anno[imname]
    vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
    keep_img = False
    for vr in vrs:
        if (vr[0] == cls_name or vr[2] == cls_name):
           if not (vr in vr_exclusions):
               keep_img = True
               break
    if keep_img:
        images_to_review.append(imname)
        annos_to_review.append(imanno)

print(f'number of images in result set: {len(images_to_review)}')

#%% X.3

# get name and annotations for a particular image
idx = 196
imname = images_to_review[idx]
imanno = annos_to_review[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.4 

# Find the *remaining subset* of images with annotations that use  
# object class 'glasses' to refer exclusivly to *drinking glasses* so that
# this *remaining subset* can be processed enmass, using a *global switch* 
# approach, to have all references to 'glasses' changed to the newly 
# introduced object class of 'drinking glass'.

imgs_to_exclude = vrdu2.glasses_exceptions_case_1 + \
                  vrdu2.glasses_exceptions_case_2 + \
                  vrdu2.glasses_exceptions_case_3 + \
                  vrdu2.glasses_exceptions_case_4

res_img_names = []
res_img_annos = []
for idx, imname in enumerate(images_to_review):
    imanno = annos_to_review[idx]
    if imname in imgs_to_exclude:
        pass
    else:
        res_img_names.append(imname)
        res_img_annos.append(imanno)

print(f'number of images in remaining subset: {len(res_img_names)}')

#%% X.5

# View images and annotations of the *remaining subset* of images thought
# to use object class 'glasses' to refer exclusivly to *drinking glasses*
# in order to verify that our analytic strategy has worked correctly.

# get name and annotations for a particular image
idx = 115
imname = res_img_names[idx]
imanno = res_img_annos[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.6

for imname in res_img_names:
    print(imname)


#%% analysis X - object class 'roof'

#%% X.1

# get all images that have a visual relationship annotation that refers
# to object class 'roof' that does not also refer to 'building' or 'tower'

cls_name = 'roof'
cls_idx = vrd_objects.index(cls_name)

b_name = 'building'
b_idx = vrd_objects.index(b_name)

t_name = 'tower'
t_idx = vrd_objects.index(t_name)

b_t_name = [b_name, t_name]

images_to_review = []
annos_to_review = []
for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    
    keep_img = False    
    if cls_idx in all_classes:
        vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
        for vr in vrs:
            if vr[0] == cls_name:
                if vr[2] in b_t_name:
                    pass
                else:
                    keep_img = True
                    break
            elif vr[2] == cls_name:
                if vr[0] in b_t_name:
                    pass
                else:
                    keep_img = True
                    break
            else:
                pass
        
    if keep_img:    
        images_to_review.append(imname)
        annos_to_review.append(imanno)

print(f'number of images found: {len(images_with_target_cls)}')

#%% X.2

# get name and annotations for a particular image
idx = 81
imname = images_to_review[idx]
imanno = annos_to_review[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X - object class 'pants'

#%% X.1

# get all images and annotations for object class 'pants'

cls_name = 'pants'

res_imgs, res_annos = vrdu.get_images_with_object_class(cls_name,
                                                        vrd_img_names, 
                                                        vrd_anno, 
                                                        vrd_objects, 
                                                        vrd_predicates)

print(f'Number of images found: {len(res_imgs)}')

#%% X.2 

# from within the result set of images, find the ones that do NOT have
# references to object class 'jeans'

jeans_idx = vrd_objects.index('jeans')

images_to_review = []
annos_to_review = []
for idx, imname in enumerate(res_imgs):
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    keep_img = True
    if jeans_idx in all_classes:
        keep_img = False
    if keep_img:
        images_to_review.append(imname)
        annos_to_review.append(imanno)

print(f'Number of images to review: {len(images_to_review)}')

#%% X.3

# get name and annotations for a particular image
idx = 230
imname = images_to_review[idx]
imanno = annos_to_review[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 3
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X - predicate 'hold'

#%% X.1

# find images where a visual relationship involves the given predicate

prd_name = 'hold'
prd_idx = vrd_predicates.index(prd_name)

images_to_review = []
annos_to_review = []
for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    if prd_idx in prd_classes:
        images_to_review.append(imname)
        annos_to_review.append(imanno)

print(f'number of images: {len(images_to_review)}')

#%% X.2

# filter out those images/annos where, for the vrs that refer to 'hold',
# the 'subject' is either 'person' or 'hand' only

person_idx = vrd_objects.index('person')
hand_idx = vrd_objects.index('hand')


images_to_review2 = []
annos_to_review2 = []
for idx, imname in enumerate(images_to_review):
    imanno = vrd_anno[imname]
    keep = False
    for vr in imanno:
        if vr['predicate'] == prd_idx:
            if ((vr['subject']['category'] == person_idx) or
                (vr['subject']['category'] == hand_idx)):
                pass
            else:
                keep = True
    if keep:
        images_to_review2.append(imname)
        annos_to_review2.append(imanno)

print(f'number of images: {len(images_to_review2)}')

#%% X.3

# get name and annotations for a particular image
idx = 19
imname = images_to_review2[idx]
imanno = annos_to_review2[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 14
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# find the images with vrs where the 'subject' and 'object' object classes
# are identical

#%% X.1

# find the images and the distinct vrs where the 'subject' and 'object'
# object classes are identical

images_to_review = []
annos_to_review = []
vrs_to_review = []
vr_counts = []
for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    keep_img = False
    for vr in imanno:
        if vr['subject']['category'] == vr['object']['category']:
            vr2 = vrdu.get_visual_relationships([vr], vrd_objects, vrd_predicates)
            if vr2[0] in vrs_to_review:
                vr_idx = vrs_to_review.index(vr2[0])
                vr_counts[vr_idx] += 1
            else:
                vrs_to_review.append(vr2[0])
                vr_counts.append(1)
            keep_img = True
    if keep_img:
        images_to_review.append(imname)
        annos_to_review.append(imanno)

print(f'number of images: {len(images_to_review)}')
print(f'number of distinct vrs: {len(vrs_to_review)}')

#%% X.2

# print the distinct vrs in descending order of frequency

vr_idx_ascending_order = np.argsort(vr_counts)

for idx in reversed(vr_idx_ascending_order):
    vr = vrs_to_review[idx]
    cnt = vr_counts[idx]
    print(vr, cnt)

#%% X.3

# find the distinct vrs NOT involving 'person'

vrs_to_review2 = []
vr_counts2 = []
for idx in reversed(vr_idx_ascending_order):
    vr = vrs_to_review[idx]
    cnt = vr_counts[idx]
    if vr[0] == 'person':
        pass
    else:
        vrs_to_review2.append(vr)
        vr_counts2.append(cnt)

print(f"number of distinct vrs not involving 'person': {len(vrs_to_review2)}")


#%% X.4

# print the distinct vrs not involving 'person' in descending order of frequency

for idx, vr in enumerate(vrs_to_review2):
    cnt = vr_counts2[idx]
    print(vr, cnt)


















