#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 11:23:27 2021

@author: dave
"""

'''
This script explores the behaviour RDFLib when used in conjunction
with OWLRL reasoning.  In particular, we explore how RDFLIB handles
duplicate triples --- ie whether it stores them or discards them.
'''

#%%

from rdflib.graph import Graph
from rdflib.term import URIRef
from rdflib.namespace import RDF
from owlrl import DeductiveClosure, OWLRL_Semantics

#%%

vrd_prefix = "http://www.semanticweb.org/dherron/ontologies/vrd_world#"

#%%

kg = Graph()

print(f'The newly instantiated graph has {len(kg)} triples')

#%%

ontology_file = '/Users/Dave/Downloads/vrd_world_ontology.owl'

kg.parse(ontology_file, format='ttl')

print(f'The KG loaded with the ontology TBox has {len(kg)} triples')


#%%

class_name = 'Umbrella'
predicate_name = 'over'

seq_num = 1000

triples = []

#
# construct triple: vrd:Umbrella1001 rdf:type vrd:Umbrella
#

class_name = 'Umbrella'
seq_num += 1

individual_name = class_name + str(seq_num)
individual_uri = URIRef(vrd_prefix + individual_name)
class_uri = URIRef(vrd_prefix + class_name)
triple = (individual_uri, RDF.type, class_uri)
triples.append(triple)
 
#
# construct triple: vrd:Person1002 rdf:type vrd:Person
#

class_name = 'Person'
seq_num += 1

individual_name = class_name + str(seq_num)
individual_uri = URIRef(vrd_prefix + individual_name)
class_uri = URIRef(vrd_prefix + class_name)
triple = (individual_uri, RDF.type, class_uri)
triples.append(triple)

#
# construct triple: vrd:Umbrella1001 vrd:over vrd:Person1002
# 

subject_name = 'Umbrella1001'
predicate_name = 'over'
object_name = 'Person1002'

subject_uri = URIRef(vrd_prefix + subject_name)
predicate_uri = URIRef(vrd_prefix + predicate_name)
object_uri = URIRef(vrd_prefix + object_name)
triple = (subject_uri, predicate_uri, object_uri)
triples.append(triple)

#
# construct triple: vrd:Umbrella1001 vrd:over vrd:Person1002  (duplicate)
# 

subject_name = 'Umbrella1001'
predicate_name = 'above'
object_name = 'Person1002'

subject_uri = URIRef(vrd_prefix + subject_name)
predicate_uri = URIRef(vrd_prefix + predicate_name)
object_uri = URIRef(vrd_prefix + object_name)
triple = (subject_uri, predicate_uri, object_uri)
triples.append(triple)

#%%

print(f'Loading {len(triples)} data triples into KG ...')

for triple in triples:
    kg.add(triple)

print(f'With data triples loaded, KG has {len(kg)} triples')

#%%

print('Running the OWLRL reasoner on the KG ...')

dc = DeductiveClosure(OWLRL_Semantics,
                      axiomatic_triples = False,
                      datatype_axioms = False)

dc.expand(kg)

print(f'After OWLRL_Semantics reasoning the KG has {len(kg)} triples')

#%%

query = """PREFIX vrd: <http://www.semanticweb.org/dherron/ontologies/vrd_world#>
           SELECT DISTINCT ?sub ?obj
           WHERE {
              ?sub vrd:above ?obj .
           }"""

qres = kg.query(query)

for row in qres:
    print(f"{row.sub} :above {row.obj}")



