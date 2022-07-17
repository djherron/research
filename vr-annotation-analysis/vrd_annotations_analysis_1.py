#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:16:59 2021

@author: dave
"""

'''
This script explores the visual relationship annotations of the images
of the VRD dataset.
'''

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import vrd_utils as vrdu

#%% get the (original) VRD data

# set path to directory in which the original VRD annotations reside
anno_dir = os.path.join('..', 'data', 'annotations_original')

# get the original VRD object class names
path = os.path.join(anno_dir, 'objects.json')
vrd_objects = vrdu.load_VRD_object_class_names(path)

# get the original VRD predicate names
path = os.path.join(anno_dir, 'predicates.json')
vrd_predicates = vrdu.load_VRD_predicate_names(path)

# get the original VRD visual relationship annotations
#path = os.path.join(anno_dir, 'annotations_train.json')
path = os.path.join(anno_dir, 'annotations_test.json')
vrd_anno = vrdu.load_VRD_image_annotations(path)

# get the original set of VRD (training or test) image names
vrd_img_names = list(vrd_anno.keys())


#%% analysis X

# number of images with visual relationship annotations in the
# customised annotations
print(f'number of images in customised annotations dictionary: {len(vrd_img_names)}')

# test set (full, original): 1000


#%% analysis X

# A first look at the visual relationship (vr) annotations for VRD images.

#%% X.1

#imname = '8013060462_4cdf330e98_b.jpg'
#imname = '7764151580_182e10b9fe_b.jpg'
#imname = '2508149623_a60b2a88eb_o.jpg'

#imname = '7421419632_a80a690fec_b.jpg'
#imname = '2340792779_b75c9d7803_b.jpg'
imname = '8427774880_31c97daac6_b.jpg'

imanno = vrd_anno[imname]

#print(imanno)
#print()

for vr in imanno:
    print(vr, '\n')

## Comments:
## 1) A VRD ('subject', 'predicate', 'object') visual relationship (vr) is 
##    represented using a Python dictionary.  The vr annotations for a
##    VRD image is a Python list of such dictionaries. 
##
## 2) Each vr has the same 5 components:
##    - 'subject' object class (represented by an integer index)
##    - 'subject' bounding box ([ymin, ymax, xmin, xmax])
##    - 'predicate' (index)    (represented by an integer index)
##    - 'object' object class  (represented by an integer index)
##    - 'object' bbox          ([ymin, ymax, xmin, xmax])

#%% X.2

# convert the visual relationships to a reader-friendly format

vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)

print(imname)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% analysis X

# Examine images and their annotated visual relationships

#%% X.0

# get the image index for a particular image name
#imname = '6279585054_1f9d450f35_b.jpg'
imname = '9919023913_194d53e585_b.jpg'
imidx = vrd_img_names.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 274  # 273 done; start at 274
imname = vrd_img_names[idx]
imanno = vrd_anno[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno, dataset='test')

#%% X.1.5 list the annotated objects

# get each unique object annotated for the image (bbox and class label)
# [Nb: this call will throw an exception if the image is found to have 
#  objects that have been assigned multiple different object classes]
bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)

print(f'number of annotated objects: {len(bboxes)}')

# display the bboxes and object classes
bboxes2 = [(lab, bb) for bb, lab in bboxes.items()]
sorted_bboxes = sorted(bboxes2)
for lab, bb in sorted_bboxes:
    classname = vrd_objects[lab]
    print(bb, lab, classname) 

#%% X.2 misc health checks

# check for duplicate VRs
res_imgs, res_annos, res_indices = vrdu.get_images_with_duplicate_vrs([imname],
                                                                      vrd_anno)
if len(res_imgs) > 0:
    print(f'WARNING, duplicate VRs: {res_indices[0]}')

# check for VRs with duplicate subject and object bboxes
res = vrdu.get_images_with_vrs_with_identical_bboxes([imname], vrd_anno)
res_imgs, res_annos, res_indices = res
if len(res_imgs) > 0:
    print(f"WARNING, vrs with identical 'sub' and 'obj' bboxes: {res_indices[0]}")

# check for VRs where the subject and object bboxes likely need to be swapped
res_imgs, res_annos, res_indices = vrdu.get_images_with_target_vr_F([imname],
                                                                    vrd_anno)
if len(res_imgs) > 0:
    print(f'WARNING, vr pair where sub/obj bboxes need swapping: {res_indices[0]}')

#%% X.3

# print all of the visual relationships for the image
# in the user-friendly format
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.4

# display image with bboxes for a particular visual relationship
vr_idx = 2
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr, dataset='test')

#%% X.5

# print all of the visual relationships for the image in raw format
for vr_idx, vr in enumerate(imanno):
    print(vr_idx, vr)

#%% analysis X

# Examine images that we couldn't examine in the analysis above because
# the images had problems (eg bboxes assigned multiple object classes)

#%% X.0

# get the image index for a particular image name
imname = '8263546381_40043c5d19_b.jpg'
imidx = vrd_img_names.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 218  # 3 done; start at 4
imname = vrd_img_names[idx]
imanno = vrd_anno[imname]

#%% X.2

# display image with bboxes for a particular visual relationship
vr_idx = 16
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr, dataset='test')

#%% X.3

# print all of the visual relationships for the image
# in the user-friendly format
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.4

# print all of the visual relationships for the image in raw format
for vr_idx, vr in enumerate(imanno):
    print(vr_idx, vr)


######################################################################


#%% analysis X

# analyse the meaning of a given object class by viewing the images
# that contain objects of that class along with the associated
# visual relationship annotations

#%% X.1

# get all images and annotations for a given object class

cls_name = 'baseball plate'

res_imgs, res_annos = vrdu.get_images_with_object_class(cls_name,
                                                        vrd_img_names, 
                                                        vrd_anno, 
                                                        vrd_objects, 
                                                        vrd_predicates)

print(f'Number of images found: {len(res_imgs)}')

#%% X.2

# get name and annotations for a particular image
idx = 5
imname = res_imgs[idx]
imanno = res_annos[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 3
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# get all the visual relationships for a given object class
vrs_for_cls = vrdu.get_all_relationships_for_object_class(cls_name,
                                                          res_imgs, res_annos,                                                     
                                                          vrd_objects, 
                                                          vrd_predicates)

print(f'number of relationship instances: {len(vrs_for_cls)}')

#%% X.4

# print visual relationships for a given object class
for i in range(300,350):
    print(vrs_for_cls[i])

#%% X.5

# get annotations for a particular image
imname = '8427774880_31c97daac6_b.jpg'
imanno = vrd_anno[imname]

#%% X.5b

# display image with bboxes for a particular visual relationship
vr_idx = 16
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)
print(imname)

#%% X.5c

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.6

# print the raw vr annotations
for idx, vr in enumerate(imanno):
    print(idx, vr)

#%% analysis X

# Analyse the predicates used with a given object class

#%% X.1

# get all distinct predicates used with a given object class

cls_name = 'sky'

res_prd = vrdu.get_distinct_predicates_for_object_class(cls_name,
                                                        vrd_img_names, 
                                                        vrd_anno,
                                                        vrd_objects, 
                                                        vrd_predicates)

print(f'Number of distinct predicates: {len(res_prd)}')

#%% X.2

# print the distinct predicates used with the given object class
for prd in res_prd:
    print(prd)


#%% analysis X

# Analyse images and annotations where a visual relationship involves a
# given subject and predicate.  And get the distinct set of object classes
# used as the object in that visual relationship.

#%% X.1

sub_name = 'person'
prd_name = 'with'

res_imgs, res_annos, res_objects = vrdu.get_images_with_target_vr_A(sub_name, 
                                                                    prd_name, 
                                                                    vrd_img_names, 
                                                                    vrd_anno, 
                                                                    vrd_objects, 
                                                                    vrd_predicates)

print(f'number of images with vr having target subject & predicate: {len(res_imgs)}')

print(f'number of distinct objects involved: {len(res_objects)}')

for obj in res_objects:
    print(obj)

#%% X.2 

for imname in res_imgs:
    print(imname)

#%% X.3

# display an image and all of its annotations

# get annotations for a particular image
imname = '4340137750_01465eff6f_b.jpg'
imanno = vrd_anno[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)



#%% analysis X

# Analyse images and annotations where a visual relationship involves a
# given predicate and object.  And get the distinct set of object classes
# used as the subject in that visual relationship.

#%% X.1

prd_name = 'above'
obj_name = 'wheel'

res_imgs, res_annos, res_subjects = vrdu.get_images_with_target_vr_D(prd_name, 
                                                                     obj_name, 
                                                                     vrd_img_names, 
                                                                     vrd_anno, 
                                                                     vrd_objects, 
                                                                     vrd_predicates)

print(f'number of images with target predicate & object: {len(res_imgs)}')

print(f'number of distinct subjects involved: {len(res_subjects)}')

for sub in res_subjects:
    print(sub)

#%% X.2 

for imname in res_imgs:
    print(imname)

#%% X.3

# display an image and all of its annotations

# get annotations for a particular image
imname = '9016049287_f8b563b6f0_o.jpg'
imanno = vrd_anno[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.ddisplay_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Find the distinct predicates that are used with a specific
# pair of 'subject' and 'object' object classes.

#%% X.1

sub_name = 'wheel'
obj_name = 'boat'

res_imgs, res_annos, res_pred = vrdu.get_images_with_target_vr_E(sub_name, 
                                                                 obj_name, 
                                                                 vrd_img_names, 
                                                                 vrd_anno, 
                                                                 vrd_objects, 
                                                                 vrd_predicates)

print(f'number of images with vr having target subject & object: {len(res_imgs)}')

print(f'number of distinct predicates involved: {len(res_pred)}')

for prd in res_pred:
    print(prd)

#%% X.2 

for imname in res_imgs:
    print(imname)


#%% analysis X

# Analyse images associated with a given visual relationship
# (subject, predicate, object)

#%% X.1 

# find images with a full target visual relationship

sub_name = 'person'
prd_name = 'stand next to'
obj_name = 'basket'

res_imgs2, res_annos2 = vrdu.get_images_with_target_vr_B(sub_name,
                                                         prd_name,
                                                         obj_name,
                                                         vrd_img_names,
                                                         vrd_anno, 
                                                         vrd_objects,
                                                         vrd_predicates)

print(f'number of images with target VR: {len(res_imgs2)}')

#%% X.2

# get name and annotations for a particular image
idx = 0
imname = res_imgs2[idx]
imanno = res_annos2[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

for imname in res_imgs2:
    print(imname)

#%% X.4

# display an image and all of its annotations

# get annotations for a particular image
imname = '85987439_0472b63475_b.jpg'
imanno = vrd_anno[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)



#%% analysis X

# Analyse the images that have a visual relationship annotation that
# uses a given predicate

#%% X.1

# find images where a visual relationship involves the given predicate

prd_name = 'touch'

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

# get name and annotations for a particular image
idx = 1
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

#%% X.3

# get all the visual relationships for a given predicate

vrs_for_prd = vrdu.get_all_relationships_for_predicate(prd_name,
                                                       images_to_review,
                                                       annos_to_review,
                                                       vrd_objects,
                                                       vrd_predicates)

print(f"number of vr instances with predicate '{prd_name}': {len(vrs_for_prd)}")

#%% X.3.1

# get the set of distinct vr instances for a given predicate

vrs = []
vr_counts = []
for im_vr_pair in vrs_for_prd:
    vr = im_vr_pair[1]
    if vr in vrs:
        vr_idx = vrs.index(vr)
        vr_counts[vr_idx] += 1
    else:
        vrs.append(vr)
        vr_counts.append(1)

print(f"number of distinct vr instances with predicate '{prd_name}': {len(vrs)}")

#%% X.3.2

# print each of the distinct vr instances for the given predicate
for idx, vr in enumerate(vrs):
    print(vr, vr_counts[idx])

#%% X.3.3

# extract the distinct 'subject' and 'object' object classes from amongst
# the set of distinct vr instances for the given predicate

distinct_subjects = []
distinct_objects = []
for vr in vrs:
    if not vr[0] in distinct_subjects:
        distinct_subjects.append(vr[0])
    if not vr[2] in distinct_objects:
        distinct_objects.append(vr[2])

print(f'number of distinct subjects: {len(distinct_subjects)}')
print()
print(f'number of distinct objects: {len(distinct_objects)}')

#%% X.3.4

print(f'distinct subjects: {distinct_subjects}')

#%% X.3.5

print(f'distinct objects: {distinct_objects}')


#%% X.4

# print visual relationships for a given predicate
for i in range(0,45):
    print(vrs[i])

#%% X.5

# display an image and all of its annotations

# get annotations for a particular image
imname = '6320506815_d815ee8b70_b.jpg'
imanno = vrd_anno[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 7
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Analyse the images with a visual relationship that uses a given predicate
# and whose subject is NOT a given object class

#%% X.1

sub_name = 'person'
prd_name = 'carry'
res_imgs, res_annos = vrdu.get_images_with_target_vr_C(sub_name, prd_name,
                                                       vrd_img_names, 
                                                       vrd_anno,
                                                       vrd_objects, 
                                                       vrd_predicates)

print(f'number of images: {len(res_imgs)}')

#%% X.2

# get name and annotations for a particular image
idx = 3
imname = res_imgs[idx]
imanno = res_annos[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

prd_idx = vrd_predicates.index(prd_name)

sub_cls_idxs = []
for imanno in res_annos:
    for vr in imanno:
        if vr['predicate'] == prd_idx:
            sub_cls_idxs.append(vr['subject']['category'])

sub_cls_idxs = list(set(sub_cls_idxs))

print(f"Subject object classes other than: '{sub_name}'")
for idx in sub_cls_idxs:
    sub_name2 = vrd_objects[idx]
    print(sub_name2)


#%% analysis X 

# analyse the distribution of the number of visual relationship (vr)
# annotations per image

#%% analysis X.1

n_imgs = len(vrd_img_names)
n_vrs_per_img = np.zeros(n_imgs)

for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    n_vrs_per_img[idx] = len(imanno)

max_vrs_per_img = int(np.max(n_vrs_per_img))
print(f'Max number of vrs per image: {max_vrs_per_img}')
## 34

print(f'Mean number of vrs per image: {np.mean(n_vrs_per_img)}')
## 7.59

print(f'Median number of vrs per image: {np.median(n_vrs_per_img)}')
## 7.59

print(f'Min number of vrs per image: {int(np.min(n_vrs_per_img))}')
## 0

#%% analysis X.2

# plot distribution of number of visual relationships (vrs) per image

bins = [idx for idx in range(max_vrs_per_img+1)]

plt.hist(n_vrs_per_img, bins)
plt.title('Distribution of number of vrs per image')

## Comments:
## 1) about 220 images have zero vr annotations; these are useless
## 2) about 200 images have only 1 vr annotation; surely many of these
##    images have more to contribute
## 3) heavy skew (long tail) to the right


#%% analysis X

# analyse the distribution of the number of distinct object classes
# referenced in the annotations of each image

#%% X.1

# get the number of object classes present in each image so we can explore
# the distribution

n_imgs = len(vrd_img_names)
n_obj_cls_per_img = np.zeros(n_imgs)
for idx, imname in enumerate(vrd_img_names):
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    n_obj_cls_per_img[idx] = len(all_classes)

#%% X.2

max_obj_cls_per_img = int(np.max(n_obj_cls_per_img))
print(f'Max number of object classes per image: {max_obj_cls_per_img}')
## 13

print(f'Mean number of object classes per image: {np.mean(n_obj_cls_per_img)}')
## 5.09

print(f'Median number of object classes per image: {np.median(n_obj_cls_per_img)}')
## 5.0

print(f'Min number of object classes per image: {int(np.min(n_obj_cls_per_img))}')
##  0

#%% X.3

# plot distribution of number of distinct object classes referenced per image

bins = [idx for idx in range(max_obj_cls_per_img+1)]

plt.hist(n_obj_cls_per_img, bins)
plt.title('Distribution of number of object classes per image')

## Comments:
## 1) a more Normal-like (Gaussian-like) distribution, except for the left
##    tail
## 2) the approximately 220 images whose annotations reference zero object
##    classes corresponds to the roughly 220 images with zero annotations
## 3) the approximately 30 images whose annotations reference only one
##    object class should be investigated


#%% analysis X

# review the images whose annotations reference zero object classes

#%% X.1

# get the names of the images whose annotations reference zero object classes

images_with_no_objects = []
for imname in vrd_img_names:
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    if len(all_classes) == 0:
      images_with_no_objects.append(imname)

print(f'Number of images with 0 annotated objects: {len(images_with_no_objects)}')

#%% X.2

# inspect the annotations for a particular image

idx = 7
imname = images_with_no_objects[idx]
imanno = vrd_anno[imname]

print(f'number of vrs: {len(imanno)}')

#%% X.3

# display image without vr annotations

vrdu.display_image(imname)

print(imname)

#%% X.4

# get the names of the images with empty annotations

images_with_empty_annotations = []
for imname in vrd_img_names:
    imanno = vrd_anno[imname]
    if len(imanno) == 0:
        images_with_empty_annotations.append(imname)

print(f'Number of images with no annotations: {len(images_with_empty_annotations)}')

#%% X.5

# are the two conditions equivalent?
# i.e., is the set of image names for images whose annotations reference
# zero object classes identical to the set of image names with zero
# vr annotations?

same = images_with_no_objects == images_with_empty_annotations
print(same)

#%% X.6

# get the names of the images with just 1 vr annotation

images_with_only_one_vr_annotation = []
for imname in vrd_img_names:
    imanno = vrd_anno[imname]
    if len(imanno) == 1:
        images_with_only_one_vr_annotation.append(imname)

print(f'Number of images with just 1 vr annotation: {len(images_with_only_one_vr_annotation)}')
# 198

#%% X.7

# get name and annotations for a particular image
idx = 20
imname = images_with_only_one_vr_annotation[idx]
imanno = vrd_anno[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# analyse the distribution of the number of predicates referenced in the
# the vr annotations per image

#%% X.1

# get the number of distinct predicates referenced per image

n_pred_per_img = []
for imname in vrd_img_names:
    imanno = vrd_anno[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    n_pred_per_img.append(len(prd_classes))

#%% X.2

max_pred_per_img = int(np.max(n_pred_per_img))
print(f'Max number of predicates per image: {max_pred_per_img}')
## 13

print(f'Mean number of predicates per image: {np.mean(n_pred_per_img)}')
## 4.3

print(f'Median number of predicates per image: {np.median(n_pred_per_img)}')
## 4.0

print(f'Min number of predicates per image: {int(np.min(n_pred_per_img))}')
## 0

#%% X.3

# plot distribution of number of distinct predicates referenced per image

bins = [idx for idx in range(max_pred_per_img+1)]

plt.hist(n_pred_per_img, bins)
plt.title('Distribution of number of predicates per image')

## Comments:
## 1) a Normal-like (Gaussian-like) distribution, except for the left tail
## 2) the approx. 220 images referencing 0 predicates corresponds to the
##    roughly 220 images with zero annotations
## 3) the roughly 300 images referencing just 1 predicate warrant
##    investigation


#%% analysis X

# analyse the distribution of image sizes (W x H)

#%% X.1

img_sizes = []
for imname in vrd_img_names:
    size = vrdu.get_image_size(imname)
    img_sizes.append(size)

#%% X.2

img_sizes_set = set(img_sizes)
print(f'Number of distinct image sizes: {len(img_sizes_set)}')

#%% X.3

# find the smallest and largest image sizes

min_width = 5000
min_height = 5000
max_width = 0
max_height = 0
min_size = [5000,5000]
max_size = [0,0]
for size in img_sizes_set:
    if size[0] > max_width:
        max_width = size[0]
    if size[1] > max_height:
        max_height = size[1]
    if size[0] < min_width:
        min_width = size[0]
    if size[1] < min_height:
        min_height = size[1]
    if size[0] >= max_size[0] and size[1] >= max_size[1]:
        max_size[0] = size[0]
        max_size[1] = size[1]
    if size[0] <= min_size[0] and size[1] <= min_size[1]:
        min_size[0] = size[0]
        min_size[1] = size[1]

print(f'min width: {min_width}')
print(f'min height: {min_height}')
print(f'max width: {max_width}')
print(f'max height: {max_height}')
print(f'max size (W x H): {max_size}')
print(f'min size (W x H): {min_size}')


#%% analysis X

# find images whose annotations contain pairs of visual relationships (vrs)
# that appear to be intended to be inverses of one another but where the
# inversion is flawed because the bounding boxes of the two respective
# objects ('subject' and 'object') are in the wrong positions and need to
# be swapped
#
# example:
# (person, wear, hat)    (hat, on, person)
# (bb_p)       (bb_h)    (bb_p)     (bb_h)
#

#%% X.1

res_imgs, res_annos, res_indices = vrdu.get_images_with_target_vr_F(vrd_img_names,
                                                                    vrd_anno)

print(f'number of images: {len(res_imgs)}')
# 12

#%% X.2

# get name and annotations for a particular image
idx = 11
imname = res_imgs[idx]
imanno = res_annos[idx]
vrpair = res_indices[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 6
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f'vr index pair: {vrpair}')

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# find images whose annotations contain duplicate visual relationships

#%% X.1

res_imgs, res_annos, res_indices = vrdu.get_images_with_duplicate_vrs(vrd_img_names,
                                                                      vrd_anno)

print(f'number of images: {len(res_imgs)}')
# 323

#%% X.2

# get name and annotations for a particular image
idx = 1
imname = res_imgs[idx]
imanno = res_annos[idx]
vrpair = res_indices[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f'vr pair: {vrpair}')

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# display an image and all of its annotations

# get annotations for a particular image
imname2 = '3469348120_a687692079_b.jpg'
imanno2 = vrd_anno[imname2]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno2[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname2, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno2, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# find all images with a vr where the 'subject' and 'object' bboxes
# are identical

#%% X.1

res = vrdu.get_images_with_vrs_with_identical_bboxes(vrd_img_names,
                                                     vrd_anno)

res_imgs, res_annos, res_indices = res

print(f'Number of images: {len(res_imgs)}')
# 98

vr_cnt = 0
for vr_indices in res_indices:
    vr_cnt += len(vr_indices)

print(f"Number of vrs where 'sub' and 'obj' bboxes are identical: {vr_cnt}")
# 105

#%% X.2

# get name and annotations for a particular image
idx = 95
imname = res_imgs[idx]
imanno = res_annos[idx]
vr_idxs = res_indices[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 6
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f"vrs with identical 'sub' and 'obj' bboxes: {vr_idxs}")

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects, vrd_predicates)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Analyse the bboxes and their object classes for image annotations

#%% X.1

#imname = '8013060462_4cdf330e98_b.jpg'
#imname = '7764151580_182e10b9fe_b.jpg'
#imname = '2547262138_07339a6f79_b.jpg'

#imname = '9282375036_2fab66f7fb_b.jpg'
imname = '8108277436_f0c3089030_b.jpg'

imanno = vrd_anno[imname]

for vr in imanno:
    print(vr, '\n')

#%% X.2

# get the unique set of bboxes and their object classes for a given image

bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)

print(len(bboxes))

print()

for k, v in bboxes.items():
    print(k, v)

#%% X.3 

# find all images that have a bbox that is assigned multiple object classes

res = vrdu.get_images_with_bboxes_having_multiple_object_classes(vrd_img_names,
                                                                 vrd_anno) 
                                   
res_imgs = res

print(f'number of images: {len(res_imgs)}') 
# 120 of the original images

#%% analysis X

# Check for degenerate bounding boxes
# All bboxes should have positive height and width

#%% X.1

# Format: [ymin, ymax, xmin, xmax]
# So we should have: ymin < ymax  and  xmin < xmax

res = vrdu.get_images_with_degenerate_bboxes(vrd_img_names, vrd_anno)
res_imgs, res_vr_idxs = res

print(f'number of images with degenerate bboxes: {len(res_imgs)}')
# 0

#%% X.2

# get name and annotations for a particular image
idx = 0
imname = res_imgs[idx]
imanno = vrd_anno[imname]
vr_idxs = res_vr_idxs[idx]

print('image:', imname)
print('vrs with bad bbox:', vr_idxs)

for idx, vr in enumerate(imanno):
    print(idx, vr)

#%%













