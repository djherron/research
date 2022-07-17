#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 06 09:16:35 2021

@author: dave
"""

'''
This module contains utility functions that support the loading of
VRD image visual relationship annotations into a knowledge graph
governed by the VRD-world ontology.

A visual relationship annotation can be described informally as a 
triple consisting of (subject, predicate, object), where a bounding
box is associated with the 'subject' object and the 'object' object.
For example, we might have (person, ride, horse).

In memory, a visual relationship is represented by a Python dictionary
with a format similar to the following pattern:

{'predicate': 35,
 'object': {'category': 43, 'bbox': [376, 804, 133, 614]},
 'subject': {'category': 0, 'bbox': [149, 495, 267, 523]}}
    
The object class and predicate categories are indicated by integers. Here
we have (0, 35, 43).  The 'subject' and 'object' categories are index 
values into the list of object class names and the 'predicate' integer
is an index into the list of predicate names. For example, object class
0 maps to 'person' and 43 maps to 'horse'. Similarly, the 
'predicate' 35 maps to predicate name 'ride'.

For bounding box specifications for the detected objects, the VRD
visual relationship annotations use the format [ymin, ymax, xmin, xmax].
'''

#%%

from rdflib.term import URIRef, Literal
#from rdflib.namespace import Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD

#%%

vrd_prefix = "http://www.semanticweb.org/dherron/ontologies/vrd_world#"

#%%

def convert_VRD_classNames_to_custom_onto_classNames(vrd_objects):
    '''
    Convert each VRD object class name to its corresponding VRD-world
    ontology class name.
    
    The VRD-world class names are all camel case, with the
    first word capitalised as well, and no spaces between words.
    '''
    
    onto_class_names = []
    
    for name in vrd_objects:
        name = name.lower()
        words = name.split()
        onto_class_name = ''
        for word in words:
            onto_class_name = onto_class_name + word.capitalize()
        onto_class_names.append(onto_class_name)

    return onto_class_names

#%%

def convert_VRD_predicateNames_to_custom_onto_propertyNames(vrd_predicates):
    '''
    Convert each VRD predicate name to its corresponding VRD-world 
    ontology object property name.
    
    The VRD-world ontology object property names are all camel case, with
    the first word uncapitalised, and no spaces between words.
    '''
    
    onto_property_names = []
    
    for name in vrd_predicates:
        name = name.lower()
        words = name.split()
        onto_property_name = ''
        for idx, word in enumerate(words):
            if idx == 0:
                onto_property_name = word
            else:
                onto_property_name = onto_property_name + word.capitalize()
        onto_property_names.append(onto_property_name)

    return onto_property_names   


#%%

def build_triples_for_image(imname, seq_numbers):
    '''
    Construct and return the triples that will represent an individual
    VRD image within a KG.
    
    Parameters:
        imname : string - the filename of a VRD image
        seq_numbers : dictionary - various integer sequence numbers
    
    Returns:
        triples : list - of RDFlib RDF triples
        subject_uri : RDFlib URIREF - the URI of the individual image
    
    Note: the seq_numbers are updated here, 'in place'; that update  
    represents an essential supplementary 'return value' to be aware of.
    '''
    
    triples = []
 
    #
    # target triple: vrd:ImageNNNNN rdf:type owl:NamedIndividual
    #
    
    class_name = 'Image'

    # get the sequence number that will uniquely identify the new
    # individual VRD image within a KG
    seq_num_key = 'image_seq_num'
    seq_numbers[seq_num_key] += 1
    seq_num = seq_numbers[seq_num_key]   
    
    # construct the triple
    individual_name = class_name + str(seq_num)
    individual_uri = URIRef(vrd_prefix + individual_name)
    triple = (individual_uri, RDF.type, OWL.NamedIndividual)
    triples.append(triple)
     
    #
    # target triple: vrd:ImageNNNNN rdf:type vrd:Image
    #

    # construct the triple
    class_uri = URIRef(vrd_prefix + class_name)
    triple = (individual_uri, RDF.type, class_uri)
    triples.append(triple)

    #
    # target triple:
    #   vrd:ImageNNNNN rdfs:label '3493152457_8dde981cc9_b.jpg'^^xsd:string
    #

    # construct the triple
    literal = Literal(imname, datatype=XSD.string)
    triple = (individual_uri, RDFS.label, literal)
    triples.append(triple)

    return triples, individual_uri


#%%

def build_triples_for_individual(cls_idx, bbox, seq_numbers, 
                                 ontoClassNames, image_uri):
    '''
    Construct and return the triples for representing a new, individual
    object within a KG.
    
    Parameters:
        cls_idx : integer - the index of the individual's VRD object class
        bbox : list of integers - the individual object's bounding box
        seq_numbers : dictionary - various integer sequence numbers
        ontoClassNames : list - ordered list of ontology class names
    
    Returns:
        triples : list - of RDFlib RDF triples
    
    Note: the seq_numbers are updated here, 'in place'; that update  
    represents an essential supplementary 'return value' to be aware of.
    
    Note: in the 'target triple' comments below, references to
    class 'Person' are meant to represent any ontology class that 
    corresponds to a VRD object class.
    '''
    
    triples = []
    
    #
    # target triple: vrd:PersonNNNNN rdf:type owl.NamedIndividual
    #
    
    class_name = ontoClassNames[cls_idx]
    
    # get the sequence number that will uniquely identify the new
    # individual VRD image within a KG
    seq_num_key = 'individual_seq_nums'
    seq_numbers[seq_num_key][cls_idx] += 1
    seq_num = seq_numbers[seq_num_key][cls_idx] 

    # construct the triple
    individual_name = class_name + str(seq_num)
    individual_uri = URIRef(vrd_prefix + individual_name)
    triple = (individual_uri, RDF.type, OWL.NamedIndividual)
    triples.append(triple)

    #
    # target triple: vrd:PersonNNNNN rdf:type vrd:Person
    #

    # construct the triple
    class_uri = URIRef(vrd_prefix + class_name)
    triple = (individual_uri, RDF.type, class_uri)
    triples.append(triple)


    #
    # target triple: vrd:ImageNNNNN vrd:hasObject vrd:PersonNNNNN
    #

    # construct the triple
    property_name = 'hasObject'
    property_uri = URIRef(vrd_prefix + property_name)
    triple = (image_uri, property_uri, individual_uri)
    triples.append(triple)

    #
    # target triple: vrd:PersonNNNNN vrd:sourceImage vrd:ImageNNNNN
    #

    # construct the triple
    property_name = 'sourceImage'
    property_uri = URIRef(vrd_prefix + property_name)
    triple = (individual_uri, property_uri, image_uri)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN rdf:type owl:NamedIndividual
    #

    class_name = 'Bbox'

    # get the sequence number that will uniquely identify the bounding
    # box for the new, individual object
    seq_num_key = 'bbox_seq_num'
    seq_numbers[seq_num_key] += 1
    seq_num = seq_numbers[seq_num_key]

    # construct the triple
    individual_bbox_name = class_name + str(seq_num)
    individual_bbox_uri = URIRef(vrd_prefix + individual_bbox_name)
    triple = (individual_bbox_uri, RDF.type, OWL.NamedIndividual)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN rdf:type vrd:Bbox
    #

    # construct the triple
    class_uri = URIRef(vrd_prefix + class_name)
    triple = (individual_bbox_uri, RDF.type, class_uri)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN vrd:hasCoordinateYmin 'NNN'^^xsd:integer
    #

    # construct the triple
    property_name = 'hasCoordinateYmin'
    property_uri = URIRef(vrd_prefix + property_name)    
    literal = Literal(bbox[0], datatype=XSD.integer)
    triple = (individual_bbox_uri, property_uri, literal)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN vrd:hasCoordinateYmax 'NNN'^^xsd:integer
    #

    # construct the triple
    property_name = 'hasCoordinateYmax'
    property_uri = URIRef(vrd_prefix + property_name)    
    literal = Literal(bbox[1], datatype=XSD.integer)
    triple = (individual_bbox_uri, property_uri, literal)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN vrd:hasCoordinateXmin 'NNN'^^xsd:integer
    #

    # construct the triple
    property_name = 'hasCoordinateXmin'
    property_uri = URIRef(vrd_prefix + property_name)    
    literal = Literal(bbox[2], datatype=XSD.integer)
    triple = (individual_bbox_uri, property_uri, literal)
    triples.append(triple)

    #
    # target triple: vrd:BboxNNNNN vrd:hasCoordinateXmax 'NNN'^^xsd:integer
    #

    # construct the triple
    property_name = 'hasCoordinateXmax'
    property_uri = URIRef(vrd_prefix + property_name)    
    literal = Literal(bbox[3], datatype=XSD.integer)
    triple = (individual_bbox_uri, property_uri, literal)
    triples.append(triple)

    #
    # target triple: vrd:PersonNNNNN vrd:hasBbox vrd:BboxNNNNN
    #

    # construct the triple
    property_name = 'hasBbox'
    property_uri = URIRef(vrd_prefix + property_name)    
    triple = (individual_uri, property_uri, individual_bbox_uri)
    triples.append(triple)

    return triples, individual_uri


#%%

def build_triple_linking_subject_to_object(subject_uri,
                                           prop_idx,
                                           object_uri,
                                           ontoPropNames): 
    '''
    Construct and return the single triple that links a 'subject'
    individual object with an 'object' individual object.
    
    Parameters:

    
    Returns:
        triples : list - of RDFlib RDF triples
    
    Note: in the 'target triple' comment below, the references to
    'Person1NNNN' and 'Person2NNNN' represent any two individuals
    belonging to any VRD object class. The use of 
    'relationshipProperty' refers to any VRD object property that
    corresponds to a VRD predicate.
    '''
    
    triples = []
    
    #
    # target triple:
    #   vrd:Person1NNNN vrd:relationshipProperty vrd:Person2NNNN
    #
    
    property_name = ontoPropNames[prop_idx]

    # construct the triple
    property_uri = URIRef(vrd_prefix + property_name)
    triple = (subject_uri, property_uri, object_uri)
    triples.append(triple)

    return triples














