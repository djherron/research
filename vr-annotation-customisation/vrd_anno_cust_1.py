#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:26:32 2021

@author: dave
"""

'''
This script performs Step 1 of the visual relationship (VR) annotations 
customisation process for the VRD dataset.

Step 1 of the process manages the definition of new object classes and
new predicates to be used in the VR annotations of the VRD dataset. It 
also applies specified adjustments to the text labels assigned to existing 
VRD object classes and predicates.

New object classes and predicates, and revised object class and predicate
names, are defined here, in Step 1, so that any and all references to these
new or amended object classes and predicates that appear in subsequent 
steps of the VR annotation customisation process will be recognised.

NOTE: Step 1 of the VR annotation customisation process is performed only
for the TRAINING set annotations, not the TEST set annotations. When 
performing the VR annotation customisation process for the TEST set VR
annotations, Step 1 should be redundant and, hence, should be skipped.
'''

#%%

import os
import vrd_utils3 as vrdu3

# Engage the VR annotation customisation configuration file for the 
# TRAINING set only!  (We should never need to run this Step 1 script
# when applying customisations to the TEST set VR annotations.)
import vrd_anno_cust_config_train as vrdcfg

#%% get the VRD object class names and predicate names

# Set the path to the directory in which the source VRD annotations data resides.
anno_dir = os.path.join('..', *vrdcfg.anno_dir)

# get an ordered tuple of the current VRD object class names
vrd_objects_path = os.path.join(anno_dir, vrdcfg.object_classes_file)
vrd_objects = vrdu3.load_VRD_object_class_names(vrd_objects_path)

# get an ordered tuple of the current VRD predicate names
vrd_predicates_path = os.path.join(anno_dir, vrdcfg.predicates_file)
vrd_predicates = vrdu3.load_VRD_predicate_names(vrd_predicates_path)

#%%

print('Step 1: processing begins ...')
print()

#%% introduce any new object class names

if len(vrdcfg.step_1_new_object_names) > 0:
    vrd_objects = list(vrd_objects) # make object class names mutable
    for name in vrdcfg.step_1_new_object_names:
        if name in vrd_objects:
            raise ValueError(f'new object class name already exists: {name}')
        vrd_objects.append(name)
    vrdu3.save_VRD_object_class_names(vrd_objects, vrd_objects_path)
    print(f'Extended object class names saved to file: {vrd_objects_path}')

#%% adjust/extend predicate names

predicates_adjusted = False
predicates_added = False

if len(vrdcfg.step_1_predicate_name_adjustments) > 0:
    vrd_predicates = list(vrd_predicates) # make predicate names mutable
    for adjustment in vrdcfg.step_1_predicate_name_adjustments:
        from_name, to_name = adjustment
        if not from_name in vrd_predicates:
            raise ValueError(f'predicate name not recognised: {from_name}')
        from_name_idx = vrd_predicates.index(from_name)
        vrd_predicates[from_name_idx] = to_name
        predicates_adjusted = True

if len(vrdcfg.step_1_new_predicate_names) > 0:
    if not predicates_adjusted:
        vrd_predicates = list(vrd_predicates) # convert tuple to list
    for name in vrdcfg.step_1_new_predicate_names:
        if name in vrd_predicates:
            raise ValueError(f'new predicate name already exists: {name}')
        vrd_predicates.append(name)
        predicates_added = True

if predicates_adjusted or predicates_added:
    vrdu3.save_VRD_predicate_names(vrd_predicates, vrd_predicates_path)
    print(f'Revised/extended predicate names saved to file: {vrd_predicates_path}')

#%%

print()
print('Step 1: processing complete')


