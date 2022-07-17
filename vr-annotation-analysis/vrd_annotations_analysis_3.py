#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:16:59 2021

@author: dave
"""

'''
This script explores the customised visual relationship annotations of the 
images of the VRD dataset.  It is designed to scrutinise the quality of the
customised annotations and verify that problems identified in the original
annotations have been corrected successfully by application of the annotation 
customisation process. That is, this script is a utility for verifying the 
quality of the customised annotations.
'''

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import vrd_utils as vrdu

#%% get the VRD data

# set path to directory in which the customised VRD annotations reside
anno_dir3 = os.path.join('..', 'data', 'annotations_customised')

# get the customised VRD object class names
path = os.path.join(anno_dir3, 'vrd_dh_objects.json')
vrd_objects3 = vrdu.load_VRD_object_class_names(path)

# get the customised VRD predicate names
path = os.path.join(anno_dir3, 'vrd_dh_predicates.json')
vrd_predicates3 = vrdu.load_VRD_predicate_names(path)

# get the customised VRD visual relationship annotations
#path = os.path.join(anno_dir3, 'vrd_dh_annotations_train.json')
path = os.path.join(anno_dir3, 'vrd_dh_annotations_test.json')
#path = os.path.join(anno_dir3, 'vrd_dh_annotations_train_trainsubset.json')
#path = os.path.join(anno_dir3, 'vrd_dh_annotations_train_testsubset.json')
vrd_anno3 = vrdu.load_VRD_image_annotations(path)

# get the customised set of VRD (training or test) image names
vrd_img_names3 = list(vrd_anno3.keys())


#%% analysis X

# number of images with visual relationship annotations in the
# customised annotations
print(f'number of images in customised annotations dictionary: {len(vrd_img_names3)}')
# training set (full): 3758
# test set: tbd
# train_trainsubset: 3000
# train_testsubset: 758


#%% analysis X

# Display an image and the bounding boxes for a particular vr
# (visual relationship). Print the annotated visual relationships as well.

#%% X.0

# get the image index for a particular image name
imname = '9919023913_194d53e585_b.jpg'
imidx = vrd_img_names3.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image, given its index
idx = 159
imname = vrd_img_names3[idx]
imanno = vrd_anno3[imname]

# display image with the subject/object bboxes for a particular vr
vr_idx = 14
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr, dataset='test')

print(imname)

#%% X.2

# print all of the visual relationships for the image in the
# user-friendly format
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# print the raw visual relationships so we can see the bbox specs
for idx, vr in enumerate(imanno):
    print(idx, vr)

#%% analysis X

# Display an image along with the bboxes for all annotated objects
# deemed to be present in the image

#%% X.0

# get the image index for a particular image name
imname = '9919023913_194d53e585_b.jpg'
imidx = vrd_img_names3.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 159
imname = vrd_img_names3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno, dataset='test')

#%% X.2

bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)

print(f'number of annotated objects: {len(bboxes)}')

# display the bboxes and object classes
bboxes2 = [(lab, bb) for bb, lab in bboxes.items()]
sorted_bboxes = sorted(bboxes2)
for lab, bb in sorted_bboxes:
    classname = vrd_objects3[lab]
    print(bb, lab, classname) 

#%% X.3

# print all of the visual relationships for the image
# in the user-friendly format
print('visual relationships:')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.4

# display image with the subject/object bboxes for a particular vr
vr_idx = 14
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr, dataset='test')

#%% X.5

# print all of the raw visual relationships so we can see the
# bbox specifications
for idx, vr in enumerate(imanno):
    print(idx, vr)


#%% analysis X

# Display an image along with the bboxes for all annotated objects
# deemed to be present in the image

#%% X.0

# get the image index for a particular image name
imname = '5390999080_7d993b0bc3_b.jpg'
imidx = vrd_img_names3.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 50
imname = vrd_img_names3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno, dataset='test')

#%% X.2

# choose whether to change the bbox format to [xmin, ymin, xmax, ymax] or not
# (nb: this is helpful if comparing ground-truth object bboxes to 
#  predicted object bboxes output by an object detection model)
chg_bbox_format = True    

# choose whether to increment the object class labels by 1
# (nb: this is helpful if comparing ground-truth annotations to 
#  predicted object class labels output by an object detection model)
chg_class_labels = True

# get each unique object annotated for the image (bbox and class label)
bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)

print(f'number of objects in image: {len(bboxes)}')

# display the bboxes and object classes
for bb, lab in bboxes.items():
    if chg_bbox_format:
        bb2 = [bb[2], bb[0], bb[3], bb[1]]
    else:
        bb2 = list(bb)
    
    if chg_class_labels:
        lab2 = lab + 1
    else:
        lab2 = lab
    
    print(bb2, lab2)

#%% X.3

# print all of the raw visual relationships so we can see the
# bbox specifications
for idx, vr in enumerate(imanno):
    print(idx, vr)

#%% X.4

# display image with the subject/object bboxes for a particular vr
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)


#%% analysis X

# Display an image along with individual VRs

#%% X.0

# get the image index for a particular image name
imname = '3987934391_ffec860739_b.jpg'
imidx = vrd_img_names3.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 60
imname = vrd_img_names3[idx]
imanno = vrd_anno3[imname]

#%% X.2

# display image with the subject/object bboxes for a particular vr
vr_idx = 10
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr, dataset='test')

#%% X.3

# print all of the visual relationships for the image
# in the user-friendly format
print('visual relationships:')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Display an image along with the bboxes for selected VRs only

#%% X.0

# get the image index for a particular image name
imname = '3987934391_ffec860739_b.jpg'
imidx = vrd_img_names3.index(imname)
print(f'image index: {imidx}')

#%% X.1

# get name and annotations for a particular image
idx = 60
imname = vrd_img_names3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno, dataset='test')

# print all of the visual relationships for the image
# in the user-friendly format
print('visual relationships:')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.2

# display image with the bboxes for selected VRs only
print(f'image name: {imname}')
vrs = [0,1,4]
vrdu.display_image_with_selected_vrs(imname, imanno, vrs)


#%% analysis X

# analyse the meaning of a given object class by viewing the images
# that contain objects of that class along with the associated
# visual relationship annotations

#%% X.1

# get all images whose annotations refer to a given object class

cls_name = 'train' 

res_imgs3, res_annos3 = vrdu.get_images_with_object_class(cls_name,
                                                          vrd_img_names3, 
                                                          vrd_anno3, 
                                                          vrd_objects3, 
                                                          vrd_predicates3)

print(f'Number of images found: {len(res_imgs3)}')

#%% X.2

# get name and annotations for a particular image
idx = 9
imname = res_imgs3[idx]
imanno = res_annos3[idx]

imidx = vrd_img_names3.index(imname)
print(f'image index in full list of image names: {imidx}')

# display image with bboxes for a particular visual relationship
vr_idx = 1
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.2b

# print all of the raw visual relationships so we can see the
# bbox specifications
for idx, vr in enumerate(imanno):
    print(idx, vr)

#%% X.3

# get all the visual relationships for a given object class
# (together with the name of the associated image)
vrs_for_cls = vrdu.get_all_relationships_for_object_class(cls_name,
                                                          res_imgs3, res_annos3,                                                     
                                                          vrd_objects3, 
                                                          vrd_predicates3)

print(f'number of relationship instances: {len(vrs_for_cls)}')

#%% X.4

# print visual relationships for a given object class
for i in range(0,23):
    print(vrs_for_cls[i])

#%% X.5

# get annotations for a particular image
imname = '8013060462_4cdf330e98_b.jpg'
imanno = vrd_anno3[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0

vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Analyse the predicates used with a given object class

#%% X.1

# get all distinct predicates used with a given object class

cls_name = 'sky'

res_prd3 = vrdu.get_distinct_predicates_for_object_class(cls_name,
                                                        vrd_img_names3, 
                                                        vrd_anno3,
                                                        vrd_objects3, 
                                                        vrd_predicates3)

print(f'Number of distinct predicates: {len(res_prd3)}')

#%% X.2

# print the distinct predicates used with the given object class
for prd in res_prd3:
    print(prd)


#%% analysis X

# Analyse images and annotations where a visual relationship involves a
# given subject and predicate.  And get the distinct set of object classes
# used as the object in that visual relationship.

#%% X.1

sub_name = 'jacket'
prd_name = 'on'

res_imgs3, res_annos3, res_objects3 = vrdu.get_images_with_target_vr_A(sub_name, 
                                                                       prd_name, 
                                                                       vrd_img_names3, 
                                                                       vrd_anno3, 
                                                                       vrd_objects3, 
                                                                       vrd_predicates3)

print(f'number of images with vr having target subject & predicate: {len(res_imgs3)}')

print(f'number of distinct objects involved: {len(res_objects3)}')

for obj in res_objects3:
    print(obj)

#%% X.2

# display an image and all of its annotations

# get name and annotations for a particular image
idx = 1      # 1
imname = res_imgs3[idx]
imanno = res_annos3[idx]

# print visual relationships
print(imname)
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

#%% X.4

# print the visual relationships in raw format
for idx2, vr in enumerate(imanno):
    print(idx2, vr)


#%% analysis X

# Analyse images and annotations where a visual relationship involves a
# given predicate and object.  And get the distinct set of object classes
# used as the subject in that visual relationship.

#%% X.1

prd_name = 'on the left of'
obj_name = 'sky'

res_imgs3, res_annos3, res_subjects3 = vrdu.get_images_with_target_vr_D(prd_name, 
                                                                        obj_name, 
                                                                        vrd_img_names3, 
                                                                        vrd_anno3, 
                                                                        vrd_objects3, 
                                                                        vrd_predicates3)

print(f'number of images with target predicate & object: {len(res_imgs3)}')

print(f'number of distinct subjects involved: {len(res_subjects3)}')

for sub in res_subjects3:
    print(sub)

#%% X.2

# display an image and all of its annotations

# get name and annotations for a particular image
idx = 0      # 4
imname = res_imgs3[idx]
imanno = res_annos3[idx]

# print visual relationships
print(imname)
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# display image with bboxes for a particular visual relationship
vr_idx = 10
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

#%% X.4

# print the visual relationships in raw format
for idx2, vr in enumerate(imanno):
    print(idx2, vr)


#%% analysis X

# Find the distinct predicates that are used with a specific
# pair of 'subject' and 'object' object classes.

#%% X.1

sub_name = 'bike'
obj_name = 'person'

res_imgs3, res_annos3, res_pred3 = vrdu.get_images_with_target_vr_E(sub_name, 
                                                                    obj_name, 
                                                                    vrd_img_names3, 
                                                                    vrd_anno3, 
                                                                    vrd_objects3, 
                                                                    vrd_predicates3)

print(f'number of images with vr having target subject & object: {len(res_imgs3)}')

print(f'number of distinct predicates involved: {len(res_pred3)}')

for prd in res_pred3:
    print(prd)

#%% X.2 

for imname in res_imgs3:
    print(imname)

#%% X.3

# display an image and all of its annotations

# get name and annotations for a particular image
idx = 0      
imname = res_imgs3[idx]
imanno = res_annos3[idx]

# print visual relationships
print(imname)
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.4

# display image with bboxes for a particular visual relationship
vr_idx = 12
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)



#%% analysis X

# Analyse images associated with a given visual relationship
# (subject, predicate, object)

#%% X.1 

# find images with a full target visual relationship

sub_name = 'person'
prd_name = 'hold'
obj_name = 'phone'

res_imgs3, res_annos3 = vrdu.get_images_with_target_vr_B(sub_name,
                                                         prd_name,
                                                         obj_name,
                                                         vrd_img_names3,
                                                         vrd_anno3, 
                                                         vrd_objects3,
                                                         vrd_predicates3)

print(f'number of images with target VR: {len(res_imgs3)}')

#%% X.2

# get name and annotations for a particular image
idx = 43
imname = res_imgs3[idx]
imanno = res_annos3[idx]

print(imname)
imidx = vrd_img_names3.index(imname)
print(f'image index in full list of image names: {imidx}')

# display image with bboxes for a particular visual relationship
vr_idx = 4
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% X.2b

for vr_idx, vr in enumerate(imanno):
    print(vr_idx, vr)

#%% X.3

for imname in res_imgs3:
    print(imname)

#%% X.4

# display an image and all of its annotations

# get annotations for a particular image
imname = '85987439_0472b63475_b.jpg'
imanno = vrd_anno3[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% analysis X

# Analyse the images that have a visual relationship annotation that
# uses a given predicate

#%% X.1

# find images where a visual relationship involves the given predicate

prd_name = 'touch'

prd_idx = vrd_predicates3.index(prd_name)

images_to_review = []
annos_to_review = []
for idx, imname in enumerate(vrd_img_names3):
    imanno = vrd_anno3[imname]
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
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# get all the visual relationships for a given predicate

vrs_for_prd3 = vrdu.get_all_relationships_for_predicate(prd_name,
                                                        images_to_review,
                                                        annos_to_review,
                                                        vrd_objects3,
                                                        vrd_predicates3)

print(f"number of vr instances with predicate '{prd_name}': {len(vrs_for_prd3)}")

#%% X.3.1

# get the set of distinct vr instances for a given predicate

vrs = []
vr_counts = []
for im_vr_pair in vrs_for_prd3:
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
imanno = vrd_anno3[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 7
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% analysis X

# Analyse the images with a visual relationship that uses a given predicate
# and whose subject is NOT a given object class

#%% X.1

sub_name = 'person'
prd_name = 'carry'
res_imgs3, res_annos3 = vrdu.get_images_with_target_vr_C(sub_name, prd_name,
                                                         vrd_img_names3, 
                                                         vrd_anno3,
                                                         vrd_objects3, 
                                                         vrd_predicates3)

print(f'number of images: {len(res_imgs3)}')

#%% X.2

# get name and annotations for a particular image
idx = 3
imname = res_imgs3[idx]
imanno = res_annos3[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print the visual relationships for the image
print(imname)
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3

# get the object classes appearing as the subject

prd_idx = vrd_predicates3.index(prd_name)

sub_cls_idxs = []
for imanno in res_annos3:
    for vr in imanno:
        if vr['predicate'] == prd_idx:
            sub_cls_idxs.append(vr['subject']['category'])

sub_cls_idxs = list(set(sub_cls_idxs))

print(f"Subject object classes other than: '{sub_name}'")
for idx in sub_cls_idxs:
    sub_name2 = vrd_objects3[idx]
    print(sub_name2)


#%% analysis X 

# analyse the distribution of the number of visual relationship (vr)
# annotations per image

#%% X.1

n_imgs = len(vrd_img_names3)
n_vrs_per_img = np.zeros(n_imgs)

for idx, imname in enumerate(vrd_img_names3):
    imanno = vrd_anno3[imname]
    n_vrs_per_img[idx] = len(imanno)

max_vrs_per_img = int(np.max(n_vrs_per_img))
print(f'Max number of vrs per image: {max_vrs_per_img}')
## train: 30
## test:

print(f'Mean number of vrs per image: {np.mean(n_vrs_per_img)}')
## train: 7.79
## test:

print(f'Median number of vrs per image: {np.median(n_vrs_per_img)}')
## train: 7.0
## test:

print(f'Min number of vrs per image: {int(np.min(n_vrs_per_img))}')
## train: 1
## test:

#%% X.2

# plot distribution of number of visual relationships (vrs) per image

bins = [idx for idx in range(max_vrs_per_img+1)]

plt.hist(n_vrs_per_img, bins)
plt.title('Distribution of number of vrs per image')


#%% analysis X

# analyse the distribution of the number of distinct object classes
# referenced in the annotations of each image

#%% X.1

# get the number of object classes present in each image so we can explore
# the distribution

n_imgs = len(vrd_img_names3)
n_obj_cls_per_img = np.zeros(n_imgs)
for idx, imname in enumerate(vrd_img_names3):
    imanno = vrd_anno3[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    n_obj_cls_per_img[idx] = len(all_classes)

#%% X.2

max_obj_cls_per_img = int(np.max(n_obj_cls_per_img))
print(f'Max number of object classes per image: {max_obj_cls_per_img}')
## train: 13
## test:

print(f'Mean number of object classes per image: {np.mean(n_obj_cls_per_img)}')
## train: 5.32
## test:

print(f'Median number of object classes per image: {np.median(n_obj_cls_per_img)}')
## train: 5.0
## test:

print(f'Min number of object classes per image: {int(np.min(n_obj_cls_per_img))}')
## train: 1
## test:

#%% X.3

# plot distribution of number of distinct object classes referenced per image

bins = [idx for idx in range(max_obj_cls_per_img+1)]

plt.hist(n_obj_cls_per_img, bins)
plt.title('Distribution of number of object classes per image')


#%% analysis X

# review images with a target number of VRs

#%% X.1

# get images with a target number of VRs

target_num = 1

images_with_only_one_vr_annotation = []
for imname in vrd_img_names3:
    imanno = vrd_anno3[imname]
    if len(imanno) == target_num:
        images_with_only_one_vr_annotation.append(imname)

print(f'Nr images with {target_num} VRs: {len(images_with_only_one_vr_annotation)}')
# 0:   0
# 1: 198
# 2: 214
# 3: 285

#%% X.3

# get name and annotations for a particular image
idx = 3
imname = images_with_only_one_vr_annotation[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

# print the visual relationships for the image
print(imname)
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# analyse the distribution of the number of predicates referenced in the
# the vr annotations per image

#%% X.1

# get the number of distinct predicates referenced per image

n_pred_per_img = []
for imname in vrd_img_names3:
    imanno = vrd_anno3[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    n_pred_per_img.append(len(prd_classes))

#%% X.2

max_pred_per_img = int(np.max(n_pred_per_img))
print(f'Max number of predicates per image: {max_pred_per_img}')
## train: 15
## test:

print(f'Mean number of predicates per image: {np.mean(n_pred_per_img)}')
## train: 4.67
## test:

print(f'Median number of predicates per image: {np.median(n_pred_per_img)}')
## train: 5.0
## test:

print(f'Min number of predicates per image: {int(np.min(n_pred_per_img))}')
## train: 1
## test:

#%% X.3

# plot distribution of number of distinct predicates referenced per image

bins = [idx for idx in range(max_pred_per_img+1)]

plt.hist(n_pred_per_img, bins)
plt.title('Distribution of number of predicates per image')


#%% analysis X

# analyse the distribution of image sizes (W x H)

#%% X.1

img_sizes = []
for imname in vrd_img_names3:
    size = vrdu.get_image_size(imname)
    img_sizes.append(size)

#%% X.2

img_sizes_set = set(img_sizes)
print(f'Number of distinct image sizes: {len(img_sizes_set)}')

#%% X.3

# find the smallest and largest image sizes

## CAUTION: this takes a few moments to run!!!!!!

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

print(f'min width: {min_width}') # 256
print(f'min height: {min_height}') # 192
print(f'max width: {max_width}') # 1280
print(f'max height: {max_height}') # 1280
print(f'max size (W x H): {max_size}') # WxH = 1280x1280
print(f'min size (W x H): {min_size}') # WxH = 256x192


#%% analysis X

# Get the unique (set of) bboxes and their object classes for a given image

#%% X.1

#imname = '8013060462_4cdf330e98_b.jpg'
#imname = '7764151580_182e10b9fe_b.jpg'
#imname = '2508149623_a60b2a88eb_o.jpg'

#imname = '9282375036_2fab66f7fb_b.jpg'
imname = '8108277436_f0c3089030_b.jpg'

imanno = vrd_anno3[imname]

#%% X.2

# get the unique set of bboxes and their object classes for a given image

bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)

print(imname)
print(f'Nr of unique objects (bboxes): {len(bboxes)}')

# print the unique objects (bboxes) and their assigned object class indices
print()
for k, v in bboxes.items():
    print(k, v)

#%% X.3

# print the VRs in raw format
for idx, vr in enumerate(imanno):
    print(idx, vr)


#%%  START OF PRIMARY VERIFICATION ANALYSES

#
#
#
#
#
#

#%% Analysis X

# find images with zero visual relationship annotations

images_with_no_objects = []
for imname in vrd_img_names3:
    imanno = vrd_anno3[imname]
    sub_classes, prd_classes, obj_classes = vrdu.get_object_classes_and_predicates(imanno)
    all_classes = set(sub_classes + obj_classes)
    if len(all_classes) == 0:
      images_with_no_objects.append(imname)

print(f'Number of images with 0 annotated objects: {len(images_with_no_objects)}')
# should be 0


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

res_imgs3, res_annos3, res_indices3 = vrdu.get_images_with_target_vr_F(vrd_img_names3,
                                                                       vrd_anno3)

print(f'number of images: {len(res_imgs3)}')
# should be 0

#%% X.2

# get name and annotations for a particular image
idx = 0
imname = res_imgs3[idx]
imanno = res_annos3[idx]
vrpair = res_indices3[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f'vr index pair: {vrpair}')

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.3 

imanno = vrd_anno3[imname]

for vr in imanno:
    print(vr, '\n')

#%% analysis X

# find images whose annotations contain duplicate visual relationships

#%% X.1

res_imgs3, res_annos3, res_indices3 = vrdu.get_images_with_duplicate_vrs(vrd_img_names3,
                                                                         vrd_anno3)

print(f'number of images: {len(res_imgs3)}')
# should be 0

#%% X.2

# get name and annotations for a particular image
idx = 0   # 
imname = res_imgs3[idx]
imanno = res_annos3[idx]
vrpairs = res_indices3[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f'vr pair: {vrpairs}')

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.2b

for idx, vr in enumerate(imanno):
    print(idx, vr)


#%% X.3

# display an image and all of its annotations

# get annotations for a particular image
imname2 = '5180178282_4e3717c82d_b.jpg'
imanno2 = vrd_anno3[imname2]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno2[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname2, vr)

# print visual relationships
vrs = vrdu.get_visual_relationships(imanno2, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# find all images with a vr where the 'subject' and 'object' bboxes
# are identical

#%% X.1

res3 = vrdu.get_images_with_vrs_with_identical_bboxes(vrd_img_names3,
                                                      vrd_anno3)

res_imgs3, res_annos3, res_indices3 = res3

print(f'Number of images: {len(res_imgs3)}')
# should be 0

vr_cnt = 0
for vr_indices in res_indices3:
    vr_cnt += len(vr_indices)

print(f"Number of vrs where 'sub' and 'obj' bboxes are identical: {vr_cnt}")
# should be 0

#%% X.2

# get name and annotations for a particular image
idx = 5
imname = res_imgs3[idx]
imanno = res_annos3[idx]
vr_idxs = res_indices3[idx]

# display image with bboxes for a particular visual relationship
vr_idx = 0
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

print(imname)

print(f"vrs with identical 'sub' and 'obj' bboxes: {vr_idxs}")

# print the visual relationships for the image
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)


#%% analysis X

# Find all images that have a bbox that has been assigned multiple
# different object classes

#%% X.1 

# find all images that have a bbox that is assigned multiple object classes

res3 = vrdu.get_images_with_bboxes_having_multiple_object_classes(vrd_img_names3,
                                                                  vrd_anno3) 
                                   
res_imgs3, res_vr_indices_with_problem_bbox, res_problem_bboxes = res3

print(f'number of images: {len(res_imgs3)}')
# should be 0

#%% X.2

# get name and annotations for a particular image
idx = 2
imname = res_imgs3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for a particular visual relationship
vr_idx = 10
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

problem_bbox = list(res_problem_bboxes[idx])
print(f'image: {imname}')
print(f'vr idx(s) with problem bbox: {res_vr_indices_with_problem_bbox[idx]}')
print(f'problem bbox: {problem_bbox}')
print()

vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)
print()

#%% X.3

# display the vrs with the problem bbox
for idx, vr in enumerate(imanno):
    if vr['subject']['bbox'] == problem_bbox or vr['object']['bbox'] == problem_bbox:
        print(vrs[idx])
        print(idx, vr, '\n')

#%% X.4

# display all vrs, in friendly form and raw form
for idx, vr in enumerate(imanno):
    print(vrs[idx])
    print(idx, vr, '\n')


#%% analysis X

# Check for degenerate bounding boxes
# All bboxes should have positive height and width

#%% X.1

# Format: [ymin, ymax, xmin, xmax]
# So we should have: ymin < ymax  and  xmin < xmax

res3 = vrdu.get_images_with_degenerate_bboxes(vrd_img_names3, vrd_anno3)
res_imgs3, res_vr_idxs3 = res3

print(f'number of images with degenerate bboxes: {len(res_imgs3)}')
# should be 0

#%% X.2

# get name and annotations for a particular image
idx = 0
imname = res_imgs3[idx]
imanno = vrd_anno3[imname]
vr_idxs3 = res_vr_idxs3[idx]

print('image:', imname)
print('vrs with bad bbox:', vr_idxs3)

for idx, vr in enumerate(imanno):
    print(idx, vr)

print()
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)
print()


#%% analysis X

# Find images whose visual relationship annotations contain bbox
# specifications that are highly similar to one another; ie that 
# have a bbox IoU (intersection over union) that exceeds a high
# threshold

#%% X.1

lower_iou_thresh = 0.25
upper_iou_thresh = 0.30
res3 = vrdu.get_images_with_highly_similar_bboxes(vrd_img_names3,
                                                  vrd_anno3,
                                                  lower_iou_thresh,
                                                  upper_iou_thresh,
                                                  obj_class_same=True)

res_imgs3, res_imgs_similar_bboxes, res_imgs_similar_bbox_ious = res3

print(f'number of images with highly similar bboxes: {len(res_imgs3)}')

# Analysis 1, with obj_class_same=False:
# iou > 0.80 and iou <= 1.00  was originally 340   done
# iou > 0.75 and iou <= 0.80  was originally  85   done
# iou > 0.70 and iou <= 0.75  was originally 106   done

# Analysis 2, with obj_class_same=True: 
# performed after all issues surfaced by Analysis 1 had been resolved
# iou > 0.90 and iou <= 1.00   number is:   3   done   3 flawed;  0 OK
# iou > 0.85 and iou <= 0.90   number is:   0
# iou > 0.80 and iou <= 0.85   number is:   0
# iou > 0.75 and iou <= 0.80   number is:   2   done   2 flawed;  0 OK
# iou > 0.70 and iou <= 0.75   number is:   3   done   0 flawed;  3 OK 
# iou > 0.65 and iou <= 0.70   number is:  24   done  24 flawed;  0 OK
# iou > 0.60 and iou <= 0.65   number is:  15   done  12 flawed;  3 OK
# iou > 0.55 and iou <= 0.60   number is:  15   done  12 flawed;  3 OK
# iou > 0.50 and iou <= 0.55   number is:  36   done  24 flawed; 12 OK
# iou > 0.45 and iou <= 0.50   number is:  30   done  15 flawed; 15 OK
# iou > 0.40 and iou <= 0.45   number is:  30   done   5 flawed; 25 OK
# iou > 0.35 and iou <= 0.40   number is:  52   done  16 flawed; 36 OK
# iou > 0.30 and iou <= 0.35   number is:  77   done  10 flawed; 67 OK

# Analysis 3, with obj_class_same=True:
# performed after all issues surfaced by Analysis 2 had been resolved;
# this is a re-doing of Analysis 2 to verify that all true issues that 
# needed resolution have been resolved successfully
# iou > 0.90 and iou <= 1.00   number is:   1          1 flawed;  0 OK
# iou > 0.85 and iou <= 0.90   number is:   0
# iou > 0.80 and iou <= 0.85   number is:   0
# iou > 0.75 and iou <= 0.80   number is:   1          1 flawed;  0 OK
# iou > 0.70 and iou <= 0.75   number is:   4          1 flawed;  3 OK       
# iou > 0.65 and iou <= 0.70   number is:   1          1 flawed;  0 OK
# iou > 0.60 and iou <= 0.65   number is:   2          0 flawed;  2 OK
# iou > 0.55 and iou <= 0.60   number is:   4          1 flawed;  3 OK
# iou > 0.50 and iou <= 0.55   number is:  13          0 flawed; 13 OK
# iou > 0.45 and iou <= 0.50   number is:  17          0 flawed; 17 OK
# iou > 0.40 and iou <= 0.45   number is:  24          0 flawed; 24 OK
# iou > 0.35 and iou <= 0.40   number is:  36          0 flawed; 36 OK
# iou > 0.30 and iou <= 0.35   number is:  66          0 flawed; 66 OK

# Analysis 4, with obj_class_same=True:
# iou > 0.25 and iou <= 0.30   number is:  72   ip     X flawed;  2 OK    

#%% X.2

# get name and annotations for a particular image
idx = 1   # 40 DONE; resume with 41   start was 0   
imname = res_imgs3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno)

print()
print('similar bboxes and their IoUs:')
sim_bboxes = res_imgs_similar_bboxes[idx]
sim_bbox_ious = res_imgs_similar_bbox_ious[idx]
for sim_bbox, iou in zip(sim_bboxes, sim_bbox_ious):
    print(sim_bbox, iou)

# get the unique set of bboxes and their object classes for a given image
bboxes = vrdu.get_bboxes_and_object_classes(imname, imanno)
print()
print(f'bboxes for {len(bboxes)} objects ...')
for k, v in bboxes.items():
    className = vrd_objects3[v]
    print(f'bbox: {k}, class: {v} {className}')

#%% X.3

# [ymin, ymax, xmin, xmax]

# for a given pair of similar bboxes, find the vrs that involve one
# or other of the two bboxes
idx2 = 0
b1 = list(sim_bboxes[idx2][0][0])
b2 = list(sim_bboxes[idx2][1][0])
vrs_with_b1 = []
vrs_with_b2 = []
for idx3, vr in enumerate(imanno):
    if (vr['subject']['bbox'] == b1 or vr['object']['bbox'] == b1):
        vrs_with_b1.append(idx3)
    if (vr['subject']['bbox'] == b2 or vr['object']['bbox'] == b2):
        vrs_with_b2.append(idx3)        

# get the visual relationships in user-friendly format
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)

# display the vrs with bbox b1
print()
print(f'vrs involving box {b1} ...\n')
for idx3, vr in enumerate(imanno):
    if idx3 in vrs_with_b1:
        print(idx3, vrs[idx3])
        print(idx3, vr)
        print()

# display the vrs with bbox b2
print()
print(f'vrs involving box {b2} ...\n')
for idx3, vr in enumerate(imanno):
    if idx3 in vrs_with_b2:
        print(idx3, vrs[idx3])
        print(idx3, vr)
        print()

#%% X.4

# display image with the subject/object bboxes for a particular vr
vr_idx = 11
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

#%% X.5

# print all of the visual relationships for the image
# in the user-friendly format
print('visual relationships:')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx, vr in enumerate(vrs):
    print(idx, vr)

#%% X.6

# print all of the raw visual relationships so we can see the
# bbox specifications
for idx, vr in enumerate(imanno):
    print(idx, vr)


#%% analysis X

# find the images that have more than 1 object of a given object class
# eg > 1 'street' object, > 1 'sky' object, etc

#%% X.1

obj_class_name = 'trees'

res3 = vrdu.get_images_with_multiple_objects_of_a_given_class(vrd_img_names3,
                                                              vrd_anno3,
                                                              vrd_objects3,
                                                              obj_class_name)

res_imgs3, res_bbox_class_tuples = res3

print(f'number of images with multiple objects of same class: {len(res_imgs3)}')

# Analysis 1: 
# street:   57    57 flawed;  0 OK
# sky:       5     5 flawed;  0 OK
# grass:     5     3 flawed;  2 OK
# mountain:  5     5 flawed;  0 OK
# trees:     7     2 flawed;  5 OK

# Analysis 2, after issues surfaced in Analysis 1 were resolved
# (should be 0 for each object class named here)
# street:    0     0 flawed;  0 OK
# sky:       0     0 flawed;  0 OK
# grass:     2     0 flawed;  2 OK
# mountain:  0     0 flawed;  0 OK
# trees:     5     0 flawed;  5 OK

#%% X.2

# get name and annotations for a particular image
idx = 7  # 1 DONE; resume with 2   start was 0   
imname = res_imgs3[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
vrdu.display_image_with_all_bboxes(imname, imanno)

print()
print(f'bboxes for object class {obj_class_name}:')
bbox_class_tuples = res_bbox_class_tuples[idx]
for bb_cls_tuple in bbox_class_tuples:
    print(bb_cls_tuple)

#%% X.3
# print all of the visual relationships for the image in user-friendly
# and raw formats
print()
print('visual relationships (user-friendly):')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx2, vr in enumerate(vrs):
    print(idx2, vr)

#%% X.4

# display image with the subject/object bboxes for a particular vr
vr_idx = 4
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

#%% X.5

# print all of the raw visual relationships so we can see the
# bbox specifications
print()
for idx2, vr in enumerate(imanno):
    print(idx2, vr)


#%% analysis X

# Find images with annotations using a specific 'subject' object class
# and predicate.  Amongst the result set, find the images where the
# inclusion ratio between the 'object' bbox and 'subject' bbox satisfies
# a specified condition.

#%% X.0

# (person, wear, Y) done
# (person, has, Y)  done
# (person, in, Y)   done 
# (person, with, Y) done
# (person, on, Y)   done
# (X, on, person)   done

#%% X.1a

sub_name = 'person'
prd_name = 'on'

res_imgs3, res_annos3, res_objects3 = vrdu.get_images_with_target_vr_A(sub_name, 
                                                                       prd_name, 
                                                                       vrd_img_names3, 
                                                                       vrd_anno3, 
                                                                       vrd_objects3, 
                                                                       vrd_predicates3)

print(f'number of images with target subject & predicate: {len(res_imgs3)}')

print(f'number of distinct objects involved: {len(res_objects3)}')

for obj in res_objects3:
    print(obj)

# NOTE: for these analyses, we typically want:   obb_within_sbb = True
# so we can calculate the inclusion ratio of the object bbox (obb) within
# the subject bbox (sbb)

#%% X.1b

prd_name = 'on'
obj_name = 'person'

res_imgs3, res_annos3, res_subjects3 = vrdu.get_images_with_target_vr_D(prd_name, 
                                                                        obj_name, 
                                                                        vrd_img_names3, 
                                                                        vrd_anno3, 
                                                                        vrd_objects3, 
                                                                        vrd_predicates3)

print(f'number of images with target predicate & object: {len(res_imgs3)}')

print(f'number of distinct subjects involved: {len(res_subjects3)}')

for sub in res_subjects3:
    print(sub)

# NOTE: for these analyses, we typically want:   obb_within_sbb = False
# so we can calculate the inclusion ratio of the subject bbox (sbb) within
# the object bbox (obb)

#%% X.2

# set these flags depending on the type of analysis being undertaken;
# they should always have opposite settings
sub_prd_analysis = False
prd_obj_analysis = True

# set to True if we want to calculate the inclusion ratio of the object
# bbox (obb) within the subject bbox (sbb); otherwise, if we want the
# inclusion ratio of the subject bbox (sbb) within the object bbox (obb),
# set to False
obb_within_sbb = False

# set inclusion ratio threshold
threshold = 0.005

images_with_target = []
image_vrs_with_problem = []
image_vrs_with_problem_incl_ratios = []

for idx, imanno in enumerate(res_annos3):
    
    vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
    vrs_with_problem = []
    vrs_with_problem_incl_ratios = []
    
    for idx2, vr in enumerate(vrs):
        process_vr = False
        if sub_prd_analysis:
            if vr[0] == sub_name and vr[1] == prd_name:
                process_vr = True
        elif prd_obj_analysis:
            if vr[1] == prd_name and vr[2] == obj_name:
                process_vr = True
        else:
            raise ValueError('analysis type flags not set properly')
        if process_vr:
            sbb = imanno[idx2]['subject']['bbox']
            obb = imanno[idx2]['object']['bbox']
            if obb_within_sbb:
                bbpair = [sbb, obb]
            else:
                bbpair = [obb, sbb]
            incl_ratio = vrdu.calc_bbox_pair_inclusion_ratio(bbpair)
            incl_ratio = round(incl_ratio, 3)
            if incl_ratio < threshold:
                vrs_with_problem.append(idx2)
                vrs_with_problem_incl_ratios.append(incl_ratio)
    
    if len(vrs_with_problem) > 0:
        imname = res_imgs3[idx]
        images_with_target.append(imname)
        image_vrs_with_problem.append(vrs_with_problem)
        image_vrs_with_problem_incl_ratios.append(vrs_with_problem_incl_ratios)

print(f'number of images with problem vrs: {len(images_with_target)}')

# Analysis 1
# threshold  count   flawed   OK   status
# (person, wear, Y)
#    < 0.50     63       28   34     done    OK usually (person, wear, skis)
# (person, has, Y)
#    < 0.50     83       19   64     done 
# (person, in, Y)
#    < 0.50     91        4   87     done    OK (person, in, vehicle or street)
# (person, with, Y)
#    < 0.50     27        1   26     done    OK varied
# (person, on, Y)
#    < 0.005    35       14   21     done    OK often (person, on, street)
# (X, on, person)
#    < 0.005    10       10    0     done

# Analysis 2 (after customisations applied)
# threshold  count   flawed   OK   status
# (person, wear, Y)
#    < 0.50     31
# (person, has,  Y)
#    < 0.50     63 
# (person, in, Y)
#    < 0.50     91
# (person, with, Y)
#    < 0.50     26
# (person, on, Y)
#    < 0.005    21
# (X, on, person)
#    < 0.005     0

#%% X.3

# get name and annotations for a particular image
idx = 6  #    4 DONE; resume at 5
imname = images_with_target[idx]
imanno = vrd_anno3[imname]

# display image with bboxes for all annotated objects
print(f'image name: {imname}')
print(f'indices of problem vrs: {image_vrs_with_problem[idx]}')
print(f'incl_ratios of problem vrs: {image_vrs_with_problem_incl_ratios[idx]}')
vrdu.display_image_with_all_bboxes(imname, imanno)

#%% X.3b

print()
print('visual relationships (user-friendly):')
vrs = vrdu.get_visual_relationships(imanno, vrd_objects3, vrd_predicates3)
for idx2, vr in enumerate(vrs):
    print(idx2, vr)

#%% X.4

# display image with the subject/object bboxes for a particular vr
vr_idx = 1
vr = imanno[vr_idx]
vrdu.display_image_with_bboxes_for_vr(imname, vr)

#%% X.5

# print all of the raw visual relationships so we can see the
# bbox specifications
print()
for idx2, vr in enumerate(imanno):
    print(idx2, vr)












