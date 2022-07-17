#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 17:58:47 2022

@author: dave
"""

'''
Utility script for testing vrd_rule_engine1.py
'''

#%% import modules

import vrd_rule_engine1 as vrdre
import vrd_utils6 as vrdu6

from rdflib.term import URIRef
from rdflib.namespace import RDF

#%%

vrd_prefix = "http://www.semanticweb.org/dherron/ontologies/vrd/vrd_dh_custom#"

#%%

def prepareTriplesForRDFLibKG():

    data = [
             ['jacket123', 'PersonalWearableItem'],
             ['person456', 'Person'],
             ['skis234', 'WinterLandSportingGood'],
             ['person457', 'Person']
           ]    

    triples = []

    for datum in data:
        individual_name = datum[0]
        individual_uri = URIRef(vrd_prefix + individual_name)
        class_uri = URIRef(vrd_prefix + datum[1])
        triple = (individual_uri, RDF.type, class_uri)
        triples.append(triple)

    return triples

#%% instantiate knowledge graph

vrdKG = vrdu6.KGWrapper(vrdu6.KGTool.RDFLIB)

#%% load knowledge graph with test data

triples = prepareTriplesForRDFLibKG()
n_triples = vrdKG.addTriplesToKG(triples)
print(f'Number of triples in KG: {n_triples}')

#%% display the data in the knowledge graph

vrdKG.displayTriplesInKG()

#%% test that we can check if a target triple is in the KG or not

existsInKG = vrdKG.checkTripleInKG('jacket123', 'PersonalWearableItem')
print(existsInKG)

existsInKG = vrdKG.checkTripleInKG('dummyIndividual', 'DummyClass')
print(existsInKG)

#%% instantiate the rule engine

# pass the RuleEngine a reference to our knowledge graph

ruleEngine = vrdre.RuleEngine(vrdKG)

#%% prepare simulated test data for RuleEngine

# image: 8054281885_ebbbfa2672_b.jpg  person,wear,jacket
# person: [159, 594, 504, 767]  [ymin, ymax, xmin, xmax]
# jacket: [199, 438, 507, 755]  [ymin, ymax, xmin, xmax]

# rule: WearRule1
data1 = {}
data1['subject'] = {}
data1['subject']['id'] = 'person456'
data1['subject']['class'] = 0       # person
data1['subject']['bbox'] = [504, 159, 767, 594]  # [xmin, ymin, xmax, ymax]
data1['object'] = {}
data1['object']['id'] = 'jacket123'
data1['object']['class'] = 19     # jacket
data1['object']['bbox'] = [507, 199, 755, 438]  # [xmin, ymin, xmax, ymax]

# image: 8054281885_ebbbfa2672_b.jpg  jacket,on,person
# person: [159, 594, 504, 767]  [ymin, ymax, xmin, xmax]
# jacket: [199, 438, 507, 755]  [ymin, ymax, xmin, xmax]

# rule: OnRuleWear1
data2 = {}
data2['subject'] = {}
data2['subject']['id'] = 'jacket123'
data2['subject']['class'] = 19     # jacket
data2['subject']['bbox'] = [507, 199, 755, 438]  # [xmin, ymin, xmax, ymax]
data2['object'] = {}
data2['object']['id'] = 'person456'
data2['object']['class'] = 0       # person
data2['object']['bbox'] = [504, 159, 767, 594]  # [xmin, ymin, xmax, ymax]


# image: 4978864715_aef3ddfac3_b.jpg  person,wear,skis
# person: [532, 770, 363, 455]  [ymin, ymax, xmin, xmax]
# skis: [710, 773, 179, 552]    [ymin, ymax, xmin, xmax]

# rule: WearRule2  
data3 = {}
data3['subject'] = {}
data3['subject']['id'] = 'person457'
data3['subject']['class'] = 0       # person
data3['subject']['bbox'] = [363, 532, 455, 770]  # [xmin, ymin, xmax, ymax]
data3['object'] = {}
data3['object']['id'] = 'skis234'
data3['object']['class'] = 77     # skis
data3['object']['bbox'] = [179, 710, 552, 773]  # [xmin, ymin, xmax, ymax]

# image: 4978864715_aef3ddfac3_b.jpg  skis,on,person
# person: [532, 770, 363, 455]  [ymin, ymax, xmin, xmax]
# skis: [710, 773, 179, 552]    [ymin, ymax, xmin, xmax]

# rule: OnRuleWear2
data4 = {}
data4['subject'] = {}
data4['subject']['id'] = 'skis234'
data4['subject']['class'] = 77     # skis
data4['subject']['bbox'] = [179, 710, 552, 773]  # [xmin, ymin, xmax, ymax]
data4['object'] = {}
data4['object']['id'] = 'person457'
data4['object']['class'] = 0       # person
data4['object']['bbox'] = [363, 532, 455, 770]  # [xmin, ymin, xmax, ymax]


#%%

data = data4

#%%

satisfiedRules = ruleEngine.execute(data)

print(satisfiedRules)

#%%





