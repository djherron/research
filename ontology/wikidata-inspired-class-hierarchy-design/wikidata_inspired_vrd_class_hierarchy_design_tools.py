#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:10:43 2021

@author: dave
"""

'''
This script explores methods and tools to help build an ontology for the
100 object classes of the VRD dataset --- an ontology that mirrors (reflects)
the ontology used within the Wikidata.org knowledge base.

For each VRD object class, we query Wikidata to find items with a label equal
to the object class name. From the result set of UIDs (Qxxxxxx) we pick the
best one.

Then we query Wikidata again, this time to find the classes in the chain of
subsumption within the Wikidata ontology, starting from our selected
matching entity UID up to the top class 'entity'.  We process the result set
to work out 1 subsumption path from 'start' up to 'entity'.

We then merge the selected subsumption chain with our iteratively evolving
KG ontology. We add the minimum number of new nodes to the ontology 
required to represent a complete subsumption path faithfully. We look to 
re-use existing ontology paths where feasible, rather than add new ones 
unnecessarily.  This step is done else, in another tool.
'''

#%%

import json
#import numpy as np
#import matplotlib.pyplot as plt
import time
import os
from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

#%%

# get an ordered list of object class names
filedir = '../phd VRD/VRD_json_dataset/'
filename = 'objects.json'
path = os.path.join(filedir, filename)
objects = json.load(open(path))

#%%

class KG_URI():
    '''
    A set of well-known KG URIs, treated as constants.
    '''
    dbpedia_uri_resource = 'http://dbpedia.org/resource/'
    dbpedia_uri_property = 'http://dbpedia.org/property/'
    dbpedia_uri = 'http://dbpedia.org/ontology/'
    
    wikidata_uri ='http://www.wikidata.org/entity/'
    
    schema_uri = 'http://schema.org/' 


#%%

class WikidataEndpoint():
    '''
    A SPARQL endpoint and prepared queries for Wikidata.
    '''
    
    def __init__(self):
        endpoint_url = "https://query.wikidata.org/sparql"
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)

    def buildQuerySubjectInstanceOfTypes(self, subject):
        # subject is an instance of which types (member of which classes)?
        # P31 == 'instance of'
        query = "SELECT DISTINCT ?uri " + \
                "WHERE { " + \
                "<" + subject + "> " + \
                "<http://www.wikidata.org/prop/direct/P31> " + \
                " ?uri . }"
        return query 
 
    def buildQueryUIDLabelLiteral(self, literal):
        # get item UID(s) for a given item label 
        query = "SELECT ?item ?itemLabel " + \
                "WHERE { " + \
                  "?item rdfs:label " + "'" + literal + "'" + "@en ." + \
                  "SERVICE wikibase:label { " + \
                      "bd:serviceParam wikibase:language " + \
                      " '[AUTO_LANGUAGE],en'. }  } " + \
                  "LIMIT 10"
        return query

    def buildQueryDescriptionForSubject(self, subject):
        query = "SELECT ?literal " + \
                "WHERE { " + \
                "<" + subject + "> " + \
                "schema:description " + \
                " ?literal . " + \
                "FILTER (lang(?literal) = 'en') }"
        return query       

    def buildQueryAlsoKnownAsForSubject(self, subject):
        # WARNING: none of these properties is correct; they all fail to 
        #          get the "Also Known As" info from an item's webpage
        # P1449 (nickname) (alias: also known as)
        # P2561 (name)  (alias: also known as)
        # P4970 (alternate names)  (alias: also known as)
        # P5973 (synonym)
        # <http://www.wikidata.org/prop/direct/P1449>
        #
        # the solution is to use: skos:altLabel
        #
        query = "SELECT ?literal " + \
                "WHERE { " + \
                "<" + subject + "> " + \
                "skos:altLabel " + \
                " ?literal . " + \
                "FILTER (lang(?literal) = 'en') }"
        return query   

    def buildQuerySubjectForAlsoKnownAs(self, literal):
        # get item UIDs for a given 'Also known as' literal
        query = "SELECT ?uri " + \
                "WHERE { " + \
                "?uri skos:altLabel " + "<" + literal + "> ." + \
                "FILTER (lang(?literal) = 'en') }"
        return query   

    def buildQuerySubclassForClass(self, superclass):
        # get the items that are subclasses of a given class
        # P279 == 'subclass of'
        query = "SELECT DISTINCT ?uri " + \
                "WHERE { " + \
                "?uri " + \
                "<http://www.wikidata.org/prop/direct/P279> " + \
                "<" + superclass + "> . }"
        return query         

    def buildQueryAllSuperclassesOfClass(self, class_uri):
        # get all of the superclasses of a given class, right up to 'entity'
        # P279 == 'subclass of'
        query = "SELECT ?class ?classLabel ?superclass ?superclassLabel " + \
                "WHERE { " + \
                    "<" + class_uri + "> wdt:P279* ?class . " + \
                    "?class wdt:P279 ?superclass . " + \
                    "SERVICE wikibase:label { " + \
                        "bd:serviceParam wikibase:language " + \
                        " '[AUTO_LANGUAGE],en'. }  } "    
        return query              

    def executeQuery(self, query, attempts=5):
        try:     
            self.sparql.setQuery(query) 
            ret = self.sparql.query()
            res = ret.convert()
            return res 
        except:    
            print("Query '%s' failed. Attempts: %s" % (query, str(attempts)))
            time.sleep(60) #to avoid limit of calls, sleep 60s
            attempts -= 1
            if attempts > 0:
                return self.executeQuery(query, attempts)
            else:
                return None

    def weLikeURI(self, uri):
        '''
        Check if we like the stem of a URI.
        '''
        weLikeThis = False
        if ( uri.startswith(KG_URI.dbpedia_uri) or \
             uri.startswith(KG_URI.wikidata_uri) or \
             uri.startswith(KG_URI.schema_uri) or \
             uri.startswith(KG_URI.dbpedia_uri_resource) or \
             uri.startswith(KG_URI.dbpedia_uri_property) ):
            weLikeThis = True
        
        return weLikeThis          

    def processResults4URI(self, results, filter_res=True):
        result_set = set()
    
        if results == None:
            print("None results")
            return result_set
            
        for result in results["results"]["bindings"]:
            #print(result)
            uri = result["uri"]["value"]
            #print(uri)
            
            if filter_res:               # filter results by URI
                if self.weLikeURI(uri):  # check if we like the URI
                    result_set.add(uri)               
            else:
                result_set.add(uri)      # return all URIs

        return result_set

    def processResults4Literal(self, results):
        result_set = set()
    
        if results == None:
            print("None results")
            return result_set
            
        for result in results["results"]["bindings"]:
            #print(result)
            value = result["literal"]["value"]
            #print(value)
            
            result_set.add(value)
        
        return result_set

    def processResults4UIDandLabel(self, results):
        result_dict = dict()
    
        if results == None:
            print("None results")
            return result_dict
            
        for result in results["results"]["bindings"]:
            #print(result)
            item_uri = result["item"]["value"]
            item_label = result["itemLabel"]["value"]
            #print(item_uri, item_label)
            
            result_dict[item_uri] = item_label
        
        return result_dict     

    def processResults4AllSuperclassesOfClass(self, results, stripUIDPrefix=False):
        result_list = []
        
        if results == None:
            print("None results")
            return result_list
        
        for result in results["results"]["bindings"]:
            #print(result)
            #print(type(result))
            #print(result.keys())
            #print()
            class_uri = result["class"]["value"]
            class_label = result["classLabel"]["value"]
            superclass_uri = result["superclass"]["value"]
            superclass_label = result["superclassLabel"]["value"]
             
            if stripUIDPrefix:
                class_uri = class_uri.split(sep='/')[4]
                superclass_uri = superclass_uri.split(sep='/')[4]

            result_list.append([class_uri, class_label, superclass_uri, superclass_label])
        
        return result_list             

    def getTypesForSubject(self, subject):
        query = self.buildQuerySubjectInstanceOfTypes(subject)
        results = self.executeQuery(query, 3)
        results = self.processResults4URI(results, filter_res=False)
        return results

    def getUIDsForLabelLiteral(self, label_literal):
        query = self.buildQueryUIDLabelLiteral(label_literal)
        results = self.executeQuery(query, 3)
        results = self.processResults4UIDandLabel(results)
        return results

    def getDescriptionsForSubject(self, subject):
        query = self.buildQueryDescriptionForSubject(subject)
        results = self.executeQuery(query, 3)
        results = self.processResults4Literal(results)
        return results        

    def getAlsoKnownAsForSubject(self, subject):
        query = self.buildQueryAlsoKnownAsForSubject(subject)
        results = self.executeQuery(query, 3)
        results = self.processResults4Literal(results)
        return results    

    def getSubjectForAlsoKnownAs(self, aka_literal):
        query = self.buildQuerySubjectForAlsoKnownAs(aka_literal)
        results = self.executeQuery(query, 3)
        results = self.processResults4URI(results)
        return results   

    def getSubclassesForClass(self, class_uri):
        query = self.buildQuerySubclassForClass(class_uri)
        results = self.executeQuery(query, 3)
        results = self.processResults4URI(results)
        return results   

    def getAllSuperclassesOfClass(self, class_uri, stripUIDPrefix=False):
        query = self.buildQueryAllSuperclassesOfClass(class_uri)
        results = self.executeQuery(query, 3)
        results = self.processResults4AllSuperclassesOfClass(results, stripUIDPrefix)
        return results   


#%% X - work outside the WikidataEndpoint class (at first)

# set endpoint url
#endpoint_url = "https://query.wikidata.org/bigdata/namespace/wdq/sparql?query={SPARQL}"
endpoint_url = "https://query.wikidata.org/sparql"

# create SPARQLWrapper stream to designated endpoint
sparql = SPARQLWrapper(endpoint_url)

# set the format for our query results
sparql.setReturnFormat(JSON)

# set subject for a query
sub = "http://www.wikidata.org/entity/Q7378"  # elephant

# compose a query: get the (english) labels of entity 'elephant'
query = "SELECT  ?label " + \
        "WHERE { " + \
             "wd:Q7378 rdfs:label ?label . " + \
             "FILTER (langMatches( lang(?label), 'EN' ) ) }"

# register the query with our sparql endpoint stream
sparql.setQuery(query)

#%% X.1 - execute the query and get raw results

ret = sparql.query()

#%% X.2 - convert the raw results to our designated format

res_dict = ret.convert()

for result in res_dict["results"]["bindings"]:
    print(result['label']['value'])

## =========================================

#%% X - work with the WikidataEndpoint class

ep = WikidataEndpoint()

#%%

# get the classes of which a given entity is an instance

sub = "http://www.wikidata.org/entity/Q7378"  # elephant
#sub = "http://www.wikidata.org/entity/Q862089"  # giraffe
types = ep.getTypesForSubject(sub)
print(len(types), types)

#%%

# get wikidata entity UIDs that have a specified label

label_literal = 'luggage'
uids = ep.getUIDsForLabelLiteral(label_literal)
print(uids.items())

#%%

# get the wikidata entity UIDs that have label values equivalent to
# the 100 object classes of the VRD dataset

uids_mstr = []    # list of dictionaries of UID:label for a given label
for i in range(len(objects)):
    obj_cls = objects[i]
    uids = ep.getUIDsForLabelLiteral(obj_cls)
    uids_mstr.append(uids)

#%%

# Get the english 'description' for a given entity

sub = "http://www.wikidata.org/entity/Q7378"  # Q7378 elephant
descs = ep.getDescriptionsForSubject(sub)
print(descs)

#%%

# Get the english 'Also known as' values for a given entity

sub = "http://www.wikidata.org/entity/Q10884"  # Q10884 tree
aka = ep.getAlsoKnownAsForSubject(sub)
print(aka)

#%%

# find entities with a specified english 'Also known as'

aka = "faucette"
uids = ep.getSubjectForAlsoKnownAs(aka)
print(uids)

#%%

# find subclasses of a given class

#class_uri = "http://www.wikidata.org/entity/Q10884"  # tree
class_uri = "http://www.wikidata.org/entity/Q862089"  # giraffe
uids = ep.getSubclassesForClass(class_uri)
print(uids)

#%% X - get the subsumption chains of an entity (sco_rel)

# for a given entity, find its subsumption chain(s) up to root class 
# 'entity' (ie find all of its parents, grandparents, etc.)

#%% X.1

def findMatchingTupleIndices(sco_rel, target, column):
    '''
    Find [class, classLabel, superclass, superclassLabel] tuples with a
    value in the specified column, {0,1,2,3}, matching the target.
    '''
    tuple_match = []
    
    for i in range(len(sco_rel)):
        if sco_rel[i][column] == target:
            tuple_match.append(i)

    return tuple_match

#%% X.2 

# display path starting from bottom (bottom-up)
def displayPathBottomUp(sco_rel, bottomup_path):
    for i in range(len(bottomup_path)):
        idx = bottomup_path[i]
        print(sco_rel[idx])
    print('*** end of output ***')

# display path starting from top (top-down)
def displayPathTopDown(sco_rel, bottomup_path):
    for i in range(len(bottomup_path)):
        j = len(bottomup_path) - i - 1
        idx = bottomup_path[j]
        print(sco_rel[idx])
    print('*** end of output ***')


#%% X.3 - get a set of 'subclass of' relations

# method 1: query Wikidata via a WikidataEndpoint 

#class_uri = "http://www.wikidata.org/entity/Q7378"  # elephant
#class_uri = "http://www.wikidata.org/entity/Q862089"  # giraffe
#class_uri = "http://www.wikidata.org/entity/Q10884"  # tree
class_uri = "http://www.wikidata.org/entity/Q144"  # dog

# get the set of 'subclass of' relations (tuples) involved in the entity's
# subsumption chain(s) leading up to class 'entity'
# - tuple: [class, classLabel, superclass, superclassLabel]
sco_rel = ep.getAllSuperclassesOfClass(class_uri, stripUIDPrefix=True)

for i in range(len(sco_rel)):
    print(sco_rel[i])    


#%% X.4 - define customised sets of sco relations (test cases)

# method 2: custom sco relations designed for particular test cases

def custom_sco_relationsA(sco_rel):
    sr = sco_rel.copy()
    sr[3] = ['Q157957', 'perennial plant', 'Q757163', 'woody plant']
    sr[5] = ['Q757163', 'woody plant', 'Q7239', 'organism']
    return sr

def custom_sco_relationsB():
    sr = []
    sr.append(['Q100', 'a', 'Q200', 'b'])
    sr.append(['Q100', 'b', 'Q200', 'c'])
    sr.append(['Q100', 'f', 'Q200', 'h'])
    sr.append(['Q100', 'c', 'Q200', 'e'])
    sr.append(['Q100', 'd', 'Q200', 'e'])
    sr.append(['Q100', 'e', 'Q200', 'f'])
    sr.append(['Q100', 'b', 'Q200', 'd'])
    sr.append(['Q100', 'e', 'Q200', 'g'])
    sr.append(['Q100', 'g', 'Q200', 'h'])
    sr.append(['Q100', 'h', 'Q200', 'i'])
    sr.append(['Q100', 'i', 'Q200', 'j'])
    sr.append(['Q100', 'j', 'Q200', 'entity'])
    return sr

def custom_sco_relationsC():
    sr = []
    sr.append(['Q100', 'a', 'Q200', 'b'])
    sr.append(['Q100', 'a', 'Q200', 'c'])
    sr.append(['Q100', 'a', 'Q200', 'd'])
    sr.append(['Q100', 'b', 'Q200', 'e'])
    sr.append(['Q100', 'c', 'Q200', 'e'])
    sr.append(['Q100', 'c', 'Q200', 'f'])
    sr.append(['Q100', 'd', 'Q200', 'f'])
    sr.append(['Q100', 'e', 'Q200', 'g'])
    sr.append(['Q100', 'f', 'Q200', 'g'])
    sr.append(['Q100', 'g', 'Q200', 'h'])
    sr.append(['Q100', 'h', 'Q200', 'entity'])
    return sr
    
def custom_sco_relationsD():
    sr = []
    sr.append(['Q100', 'a', 'Q200', 'b'])
    sr.append(['Q100', 'a', 'Q200', 'c'])
    sr.append(['Q100', 'a', 'Q200', 'd'])
    sr.append(['Q100', 'b', 'Q200', 'e'])
    sr.append(['Q100', 'c', 'Q200', 'e'])
    sr.append(['Q100', 'c', 'Q200', 'f'])
    sr.append(['Q100', 'c', 'Q200', 'g'])
    sr.append(['Q100', 'd', 'Q200', 'g'])
    sr.append(['Q100', 'e', 'Q200', 'h'])
    sr.append(['Q100', 'f', 'Q200', 'h']) 
    sr.append(['Q100', 'f', 'Q200', 'i'])
    sr.append(['Q100', 'g', 'Q200', 'i'])
    sr.append(['Q100', 'h', 'Q200', 'j'])
    sr.append(['Q100', 'i', 'Q200', 'j'])
    sr.append(['Q100', 'j', 'Q200', 'k'])
    sr.append(['Q100', 'k', 'Q200', 'entity'])
    return sr

#%% X.5 - get sco relations from a .csv file

# method 3: get sco relations from a .csv file
#           (where the .csv file is a download of the query results 
#            obtained via Wikidata's online query service at
#            https://query.wikidata.org/)
#
# In other words, if Method 1 isn't working for some reason, we can run the
# same query in Wikidata's online query service and get the same results.
# But then we have to read the .csv and convert the sco relations into our
# preferred internal format (a list of lists).

# function to read sco relations from 
def get_sco_relations_from_csv(filepath):
    df = pd.read_csv(filepath)
    sco_rel = []
    for i in range(len(df)):
        rel = list(df.iloc[i,:])
        rel[0] = rel[0].split(sep='/')[4] # extract UID Qxxx from url
        rel[2] = rel[2].split(sep='/')[4] # extract UID Qxxx from url
        sco_rel.append(rel)
    return sco_rel

#%% X.6 - select a sco relations dataset with which to work

# method 1: use data drawn from Wikidata by a SPARQL query
#sco_rel2 = sco_rel.copy()

# method 2: use custom data prepared for a test case 
#sco_rel2 = custom_sco_relationsA(sco_rel)
#sco_rel2 = custom_sco_relationsB()
#sco_rel2 = custom_sco_relationsC()
#sco_rel2 = custom_sco_relationsD()

# method 3: use data from a .csv file
filepath = '~/Downloads/wd-query-results-for-vrd-obj-007-chair.csv'
sco_rel2 = get_sco_relations_from_csv(filepath)
#df = get_sco_relations_from_csv(filepath)

#%%

for rel in sco_rel2:
    print(rel)

#%% X - find one subsumption chain (path)

# method: bottom up - single path
#
# nb: multiple subsumption paths may exist amongst the results. This method
#     returns only the first path found and ignores any others which may
#     exist.

bottomup_path = []
#target = 'elephant'  # Q7378    (start at the bottom of the subsumption chain)
#target = 'giraffe'   # Q862089
target = 'person'
for i in range(len(sco_rel2)):
    tuple_match = findMatchingTupleIndices(sco_rel2, target, 1)
    if len(tuple_match) > 0:
        idx = tuple_match[0]       # take the first match only
        bottomup_path.append(idx)
        target = sco_rel2[idx][3]       # follow the first match

print(bottomup_path)


#%% X.1 - display one path bottom-up and top-down

displayPathBottomUp(sco_rel2, bottomup_path)

print()

displayPathTopDown(sco_rel2, bottomup_path)




#%% X - discover all subsumption chains (paths to the top)


#%% X.1 - get raw (encoded) subsumption chains for an entity

def climbSubsumptionChain(sco_rel, target, sschains, sschains_aux):
    '''
    Discover all 'subclass of' subsumption chains (paths) from a given
    entity (class A) up to the top of the Wikidata class hierachy, which
    is class 'entity'.  Do this recursively.

    Parameters
    ----------
    sco_rel : list of lists
        Each list represents a 'subclass of' (sco) relation expressing that
        :classA :subClassOf :classB 
    target : string
        A string token representing the UID (Qxxx) of a class --- the class 
        at the bottom of all subsumption chains expressed in the collection
        of sco relations.
    sschains : list
        A list of integers encoding one or more subsumption chains.
        This is recursive state that is updated with each call to this
        recursive function.
    sschains_aux : list of string tokens
        Auxiliary information essential for decoding the list of integers
        into complete and correct multiple subsumption chains.
        This is recursive state that is updated with each call to this
        recursive function.
        
    Returns
    -------
    None
    
    All information gathered is returned via the updates applied to the
    mutable recursive state, so passing data via the 'return' statement
    itself isn't needed.
    
    Limitations
    -----------
    A given class, class A, could have any number of direct superclasses; 
    that is, it could have, say, K direct 'subclass of' relations. This
    version of this function supports a maximum of K=3.  If a certain class
    has more than 3 direct superclasses, only the first 3 are recognised,
    and any others are ignored.
    '''

    # base case: we've reached the top of the class hierarchy
    if target == 'Q35120':  # class 'entity'
        sschains.append(-1) # signal the end of one subsumption chain
        sschains_aux.append('top')
        #print(sschains)
        return None
    
    # recursive case: take another step up the current subsumption chain
    else:
        tuple_match = findMatchingTupleIndices(sco_rel, target, 0)
        nmatch = len(tuple_match)
        if nmatch == 0:
            print(f'Problem: no match for "{target}"')
        if nmatch > 3:
            nmatch = 3
        for j in range(nmatch):
            idx = tuple_match[j]
            sschains.append(idx)
            token = str(j+1) + ' of ' + str(nmatch)
            sschains_aux.append(token)
            #print(sschains)
            #print(sschains_aux)
            target = sco_rel[idx][2]
            climbSubsumptionChain(sco_rel, target, sschains, sschains_aux)

#%% X.2 - count the number of subsumption chains

def countChains(sschains):
    nchains = 0
    for token in sschains:
        if token == 'top':
            nchains += 1
    return nchains


#%% X.3 - decode the raw subsumption chain data to extract proper chains

def extractSSChains(sschains_raw, sschains_aux):
    '''
    Extract clean representations of each unique subsumption chain
    present in a raw subsumption chains data object for a given entity.
    A subsumption chain is path composed of 'subclass of' relations from
    a given entity up to the top of the Wikidata class hierarchy, at which
    is class 'entity'.
    
    Parameters
    ----------
    sschains_raw : list of integers
        One list of integers that encodes 1 or more subsumption
        chains from class A up to class 'entity', using 'subclass of'
        declarations only.
    sschains_aux : list of string tokens
        Auxiliary information essential for decoding the raw sschains
        data in order to recover clear representations of the 
        individual unique subsumption chains.

    Returns
    -------
    sschains : list of lists of integers
        A collection of subsumption chains, each represented as a list
        of integers, where the integer indices represent particular 
        'subclass of' relations in a set of such. Each chain expresses a
        unique path from class A up to the top of the class hierarchy,
        which, in Wikidata, is class 'entity', using 'subclass of'
        relations only.
        
    Version history
    ---------------
    v 1.1 support ternary cases (class can be subclass of 3 superclasses)
    v 1.0 support binary cases (class can be subclass of 2 superclasses)
    '''

    # indices of open 'open bracket' tokens, managed as a stack
    openbracket_binary = []
    openbracket_ternary = []

    # work on a copy, since we modify this data heavily
    sschains_raw_cpy = sschains_raw.copy()
    
    # container for refined subsumption chains
    sschains = []
    
    for idx, token in enumerate(sschains_aux):
        if token == '1 of 1':
            pass
        elif token == '1 of 2':   # open bracket (
            # push idx of ( onto top of binary stack
            openbracket_binary.append(idx)   
        elif token == '2 of 2':   # close bracket )
            # pop the top of the binary stack to get the idx of the matching
            # binary open bracket token
            bidx = openbracket_binary.pop()  
            # zap range of obsolete data
            # (from matched open bracket up to this close bracket)
            for i in range(bidx, idx):   
                sschains_raw_cpy[i] = -1
        elif token == '1 of 3':   # open ternary bracket (
            # push idx of ( onto top of ternary stack
            openbracket_ternary.append(idx)          
        elif token == '2 of 3':   # middle ternary dot * 
            bidx = openbracket_ternary.pop()  # pop open bracket idx
            for i in range(bidx, idx):        # zap spent data
                sschains_raw_cpy[i] = -1
            openbracket_ternary.append(idx)    # push middle dot idx             
        elif token == '3 of 3':   # close ternary bracket )
            bidx = openbracket_ternary.pop()  # pop middle dot idx
            for i in range(bidx, idx):        # zap spent data
                sschains_raw_cpy[i] = -1       
        elif token == 'top':
            # remove all the -1s and the resulting sequence of integers
            # represents a unique subsumption chain
            chain = [ idx2 for idx2 in sschains_raw_cpy[0:idx] if idx2 != -1 ]
            sschains.append(chain)
        else:
            raise ValueError('unexpected token')
    
    return sschains


#%%

# method 3: use data from a .csv file
filepath = '~/Downloads/wd-query-results-for-vrd-obj-041-dog.csv'
sco_rel2 = get_sco_relations_from_csv(filepath)

#%% X.4 - get the bottom-up subsumption chains in raw, encoded format

bottom_entity = 'Q144'  

#bottom_entity = 'tree'  # works
#bottom_entity = 'a'

sschains_raw = []
sschains_aux = []
climbSubsumptionChain(sco_rel2, bottom_entity, sschains_raw, sschains_aux)

print()
print(f'sschains: {sschains_raw}')
print()
print(f'sschains_aux: {sschains_aux}')


#%% X.5 - extract clear chains from the raw (encoded) chains

chains = extractSSChains(sschains_raw, sschains_aux)

print(f'number of chains: {len(chains)}')

print()
for idx, chain in enumerate(chains):
    print(idx, chain)

print()
for idx, rel in enumerate(sco_rel2):
    print(idx, rel)

#%% X.6 - render a chain using its sco relations rather than their indices

# print the sequence of 'subclass of' relations in a given subsumption chain,
# and do it bottom-up: from class A up to class 'entity'.
chain_idx = 0
chain = chains[chain_idx]
for rel_idx in chain:
    print(sco_rel2[rel_idx])


#%% X analyse issue of: Problem: no match for "Q733541"

# Why does it happen?
# - The issue arises in the execution of the SPARQL query that retrieves
#   all of the 'subclass of' relations involved in all the subsumption chians
#   for a given class.  For some reason, the 'subclass of' relation
#   <:Q733541 (consequence) :subclassOf :Q408386 (inference)> is NOT
#   in the query result set.  The reason for this is unknown.  There does
#   not appear to be anything we can do to investigate further 'why' this
#   happens during query execution, and neither is there anything we can do
#   about it.  We have to live with this issues. 
#
# test cases:
# Q22676 - 27 - shoe   3 occurrences of the issue
# Q1064858 - 29 - desk   4 occurrences of the issue
# Q2741056 - 30 - cabinet    6 occurrences of the issue
# Q1420266 - 31 - counter    2 occurrences of the issue
# Q144 - 41 - dog         1 occurrence
# Q223269 - 58 - shorts   1 occurrence

# Also: Problem: no match for "Q422742"
#
# The reason for this issue seems clear: entity Q422742 has no 'parent'
# class defined. That is, it has no 'subclass of' relation to anything.
# This is clearly a case of 'bad data' within Wikidata. For entity
# Q2355817 (plant life-form), it has
# <:Q2355817 :subclassOf :Q422742> but entity Q422742 has no parent class
# defined.
#
# test cases:
# Q42295 - 48 - bush   1 occurrence

