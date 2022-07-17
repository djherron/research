#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 20:09:22 2021

@author: dave
"""

'''
This script loads VRD image visual relationship annotations into a knowledge
graph governed by the VRD-world OWL ontology.
'''

#%%

from rdflib import Graph
from rdflib.term import URIRef, Literal

from owlrl import DeductiveClosure, OWLRL_Semantics
import copy

import os
import vrd_utils as vrdu
import vrd_utils4 as vrdu4

#%%

ontology_file = '/Users/Dave/Downloads/vrd_world_ontology.owl'

kg = Graph()

kg.parse(ontology_file, format='ttl')

#%%

print(f'The newly loaded graph has {len(kg)} triples')

#%% reason with Python package 'owlrl'

# The 'owlrl' package implements the OWL 2 RL profile as well as 
# basic RDFS inference, on top of RDFLib.  One can choose to use
# OWL 2 RL reasoning, RDFS reasoning or a combination of the two.

# The DeductiveClosure class is the main entry point to the package.
# This augments the incoming graph with all triples that the designated
# rule set permits (ie it computes the 'deductive closure' of the graph).

# The first argument to DeductiveClosure specifies the inference semantics
# you wish to have applied to your graph.  The available options and
# their meaning are as follows:
# OWLRL_Semantics  - requests OWL RL inference semantics
# RDFS_Semantics  - requests basic RDFS inference semantics
# RDFS_OWLRL_Semantics - requests combined OWL RL / RDFS inference semantics 

# The OWL 2 RL inference semantics rules/patterns are specified here:
# https://www.w3.org/TR/owl2-profiles/#OWL_2_RL
# OWL 2 RL is a syntactic subset of OWL 2 "amenable to implementation
# using rule-based technologies".  The OWL 2 RL profile restricts the
# way in which certain OWL constructs can be used. The restrictions are
# constraints on the expressions that can be used in particular syntactic
# positions of an RDF triple when that triple involves a particular OWL/RDFS
# constructs.

# The RDF 1.1 Semantics specification (which includes RDFS):
# https://www.w3.org/TR/rdf11-mt/#rdfs-entailment
# See section 9.2 for the RDFS entailment rules/patterns.

kg2 = copy.deepcopy(kg)

dc = DeductiveClosure(OWLRL_Semantics,
                      axiomatic_triples = False,
                      datatype_axioms = False)

dc.expand(kg2)

print(f'After reasoning the graph has {len(kg2)} triples')

#%%

filename = 'vrd_world_tbox_abox_owlrl_1.ttl'

print(f'Writing augmented graph to file: {filename}')

kg2.serialize(destination = filename, format = 'ttl')


#%% get the (original) VRD data

# Set the path to the directory in which the source VRD annotations data resides.
anno_dir = os.path.join('.', 'data', 'vrd_customised_anno_01', 'VRD_json_dataset')

# get an ordered tuple of the VRD object class names
vrd_objects_path = os.path.join(anno_dir, 'vrd_dh_objects.json')
vrd_objects = vrdu.load_VRD_object_class_names(vrd_objects_path)

# get an ordered tuple of the VRD predicate names
vrd_predicates_path = os.path.join(anno_dir, 'vrd_dh_predicates.json')
vrd_predicates = vrdu.load_VRD_predicate_names(vrd_predicates_path)

# get the VRD image annotations
vrd_annotations_path = os.path.join(anno_dir, 'vrd_dh_annotations_train.json')
vrd_anno = vrdu.load_VRD_image_annotations(vrd_annotations_path)

# get a list of the VRD image names from the annotations dictionary
vrd_img_names = list(vrd_anno.keys())

#%% 

ontoClassNames = vrdu4.convert_VRD_classNames_to_custom_onto_classNames(vrd_objects)

ontoPropNames = vrdu4.convert_VRD_predicateNames_to_custom_onto_propertyNames(vrd_predicates)

#%%

# initialise the sequence numbers to be used to generate unique
# names for images, individual objects and bounding boxes

start_seq_num = 10000
individual_seq_nums = [start_seq_num for name in ontoClassNames]
image_seq_num = start_seq_num
bbox_seq_num = start_seq_num

seq_numbers = {'individual_seq_nums': individual_seq_nums,
               'image_seq_num': image_seq_num,
               'bbox_seq_num': bbox_seq_num}

#%%

# create the vrd namespace we need 
#vrd = Namespace("http://www.semanticweb.org/dherron/ontologies/vrd/vrd_dh_custom#")

# bind our prefixes to 
#kg.bind("vrd", vrd)
#kg.bind("rdf", RDF)
#kg.bind("rdfs", RDFS)
#kg.bind("xsd", XSD)

#%%

imname = '3493152457_8dde981cc9_b.jpg'

# load image into KG
triples, image_uri = vrdu4.build_triples_for_image(imname, seq_numbers)
for triple in triples:
     print(triple)
     print()

#%%

kg4 = Graph()

print(f'num triples: {len(kg4)}')

#%%

for triple in triples:
     kg4.add(triple)

for triple in triples:
     kg4.add(triple)

print(f'num triples: {len(kg4)}')

#%%

kg4.add(triples[1])

kg4.add(triples[1])

print(f'num triples: {len(kg4)}')

#%%

for s, p, o in kg4:
    print(s.n3(), p.n3(), o.n3())
    print()


#%%

ymin = Literal(524)


#%%

kg3 = Graph()

#vrd_img_names2 = ['3493152457_8dde981cc9_b.jpg']
#vrd_img_names2 = ['7764151580_182e10b9fe_b.jpg']
vrd_img_names2 = ['8013060462_4cdf330e98_b.jpg']
#vrd_img_names2 = ['3493152457_8dde981cc9_b.jpg',
#                  '7764151580_182e10b9fe_b.jpg']
#vrd_img_names2 = ['3493152457_8dde981cc9_b.jpg',
#                  '7764151580_182e10b9fe_b.jpg',
#                  '8013060462_4cdf330e98_b.jpg']




#%%

for s, p, o in kg3:
    print(s.n3(), p.n3(), o.n3())

#%%

image_cnt = 0
total_triple_cnt = 0

for imname in vrd_img_names2:
    
    imanno = vrd_anno[imname]
    image_objects = {}
    image_cnt += 1
    image_triple_cnt = 0
    
    print(f'\nprocessing image: {imname}')
    
    # load image into KG
    results = vrdu4.build_triples_for_image(imname, seq_numbers)
    triples, image_uri = results
    image_triple_cnt += len(triples)
    #print(f'num triples: {len(triples)}')
    for triple in triples:
        kg3.add(triple)

    for vr in imanno:
        
        #print(f'processing vr: {vr}')
        vr_triple_cnt = 0
        
        sub_idx = vr['subject']['category']
        sub_bbox = vr['subject']['bbox']
        sub_bbox_tuple = tuple(sub_bbox)
        prd_idx = vr['predicate']
        obj_idx = vr['object']['category']
        obj_bbox = vr['object']['bbox']
        obj_bbox_tuple = tuple(obj_bbox)
        
        # process the individual relating to the 'subject' object class
        if sub_bbox_tuple in image_objects:
            subject_uri = image_objects[sub_bbox_tuple]
            triple_cnt = 0
        else:
            results = vrdu4.build_triples_for_individual(sub_idx,
                                                         sub_bbox,
                                                         seq_numbers,
                                                         ontoClassNames,
                                                         image_uri)
            triples, subject_uri = results
            triple_cnt = len(triples)
            #print(f'num subject triples: {len(triples)}')
            image_objects[sub_bbox_tuple] = subject_uri
            for triple in triples:
                kg3.add(triple)
        vr_triple_cnt += triple_cnt
        
        # process the individual relating to the 'object' object class
        if obj_bbox_tuple in image_objects:
            object_uri = image_objects[obj_bbox_tuple]
            triple_cnt = 0
        else:
            results = vrdu4.build_triples_for_individual(obj_idx,
                                                         obj_bbox,
                                                         seq_numbers,
                                                         ontoClassNames,
                                                         image_uri)
            triples, object_uri = results
            triple_cnt = len(triples)
            #print(f'num object triples: {len(triples)}')
            image_objects[obj_bbox_tuple] = object_uri
            for triple in triples:
                kg3.add(triple)
        vr_triple_cnt += triple_cnt
        
        # link the 'subject' individual to the 'object' individual
        triples = vrdu4.build_triple_linking_subject_to_object(subject_uri,
                                                               prd_idx,
                                                               object_uri,
                                                               ontoPropNames)
        vr_triple_cnt += len(triples)
        #print(f'num link triples: {len(triples)}')
        for triple in triples:
            kg3.add(triple)
        
        image_triple_cnt += vr_triple_cnt
    
    
    print(f'triples loaded to KG for image: {image_triple_cnt}')
    total_triple_cnt += image_triple_cnt


print(f'\nNumber of images processed: {image_cnt}')

print(f'\nTotal triples loaded to KG: {total_triple_cnt}')

#%%

print(f'After loading abox data, the graph has {len(kg3)} triples')

#%%

cnt = 0
for s, p, o in kg3:
    cnt += 1
    print(s.n3(), p.n3(), o.n3())
    print()

print(f'num triples: {cnt}')

#%%

filename = 'vrd_world_tbox_abox.ttl'
#print(f'serialising kg3 to file: {filename}')
#kg.serialize(destination='vrd_world_tbox_abox.ttl', format='ttl')












