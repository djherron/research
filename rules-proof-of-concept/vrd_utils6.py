#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:45:48 2022

@author: dave
"""

'''
Classes and utility functions relating to the accessing of knowledge graphs.

Currently supported knowledge graph tools:
* RDFLib  (implemented)
* GraphDB (TODO)
'''

#%%

from rdflib.graph import Graph
from enum import Enum

#%%

vrd_prefix = "<http://www.semanticweb.org/dherron/ontologies/vrd/vrd_dh_custom#>"

#%%

class KGTool(Enum):
    RDFLIB = 1
    GRAPHDB = 2

#%%

class KGWrapper:
    
    def __init__(self, kgTool):    
        if kgTool == KGTool.RDFLIB:
            self.kg = Graph()
            print(f'New RDFLib KG instantiated; KG has {len(self.kg)} triples')
        elif kgTool == KGTool.GRAPHDB:
            raise ValueError('GraphDB KG not yet implemented')
        else:
            raise ValueError('kgTool type not recognised')
        self.kgTool = kgTool

    def getNumberOfTriplesInKG(self):
        if self.kgTool == KGTool.RDFLIB:
            n_triples = len(self.kg)
        elif self.kgTool == KGTool.GRAPHDB:
            raise ValueError('GraphDB KG not yet implemented')
        else:
            raise ValueError('kgTool type not recognised')         
        return n_triples

    def addTriplesToRDFLibKG(self, triples):
        for triple in triples:
            self.kg.add(triple)        
        return len(triples)
    
    def addTriplesToKG(self, triples):
        if self.kgTool == KGTool.RDFLIB:
            n_triples = self.addTriplesToRDFLibKG(triples)
        elif self.kgTool == KGTool.GRAPHDB:
            raise ValueError('GraphDB KG not yet implemented')
        else:
            raise ValueError('kgTool type not recognised')        
        return n_triples

    def displayTriplesInKG(self, limit=20):
        if self.kgTool == KGTool.RDFLIB:
            cnt = 0
            for s, p, o in self.kg:
                cnt += 1
                if cnt <= limit:
                    print(cnt, (s.n3(), p.n3(), o.n3()))
                else:
                    break                
        elif self.kgTool == KGTool.GRAPHDB:
            raise ValueError('GraphDB KG not yet implemented')
        else:
            raise ValueError('kgTool type not recognised')
        return None

    def buildASKSparqlQuery(self, subj, obj):   
        prefix = "PREFIX vrd: " + vrd_prefix
        part1 = "ASK WHERE { "
        triple_subject = "vrd:" + subj
        triple_object = "vrd:" + obj
        part2 = triple_subject + " rdf:type " + triple_object + " }"  
        query = prefix + ' ' + part1 + ' ' + part2 
        return query
     
    def executeASKSparqlQuery(self, query):    
        result = None
        if self.kgTool == KGTool.RDFLIB:
            qres = self.kg.query(query)
            if len(qres) != 1:
                raise ValueError('response to kgTool query not recognised')
            for response in qres:
                result = response
        elif self.kgTool == KGTool.GRAPHDB:
            raise ValueError('GraphDB KG not yet implemented')
        else:
            raise ValueError('kgTool not recognised in executeSparqlQuery()')        
        return result    

    def checkTripleInKG(self, subj, obj):
        query = self.buildASKSparqlQuery(subj, obj)
        result = self.executeASKSparqlQuery(query)       
        return result

#%%



