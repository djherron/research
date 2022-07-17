#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 19:49:57 2022

@author: dave
"""

'''
An initial exploration of implementing a common-sense knowledge rule
engine for use as part of a NN-KG-I visual relationship prediction
system.

conceptual rule structure:
* rule ::= head :- body
* head ::= <A, vrd:predicate, B>
* body ::= condition | condition, condition | condition, condition, condition | ...
* condition ::= rdf_triple_template | function_condition
* rdf_triple_template ::= <A, rdf:type, #className> | <B rdf:type, #className>
* function_condition ::= functionCall arithOp value | functionCall_expression arithOp value
* A, B ::= names of individuals (unique names of particular objects detected in particular images; individual members of OWL classes); the individuals correspond to unique objects detected by the object detector in a particular image; each detected individual (A, B) has a predicted bbox (Abb, Bbb) and a predicted class (Ac, Bc)
* ',' ::= a comma between conditions in a rule body denotes 'conjunction'

example: rule for cases when predicate 'on' is a valid inverse of predicate 'wear'
* a rule for when it is likely to be valid to use 'on' as an inverse of 'wear'
* rule: <A, vrd:on, B> :- <A, rdf:type, vrd:WearableItem>, 
                          <B, rdf:type, vrd:Person>,
                          ir(Abb,Bbb) approx 1,
                          area(Abb)/area(Bbb) << 1

rule processing
* for the body of a rule, process each condition in sequence, from left to right
* if the body condition is an rdf_triple_template:
  - form RDF triple <A, rdf:type, Ac> and insert it into the KG
  - then instantiate the triple template with the appropriate individual name 
    and query the KG to see whether that triple now exists in the KG or not; 
    if YES, the condition is satisfied, so continue processing the body; 
    if NO, the rule body is not satisfied; stop processing the rule and move 
    to the next rule in the rule base (if there is one)
* if the body condition is a function_condition:
  - call the function and determine the outcome of the condition
  - if the condition is satisfied, continue processing the rule body; 
    else, the rule body is not satisfied; stop processing the rule and move 
    to the next rule in the rule base (if there is one)
* if the rule body of a rule is satisfied (ie all conditions are 
  satisfied/true), then the instantiated 'head' of the rule (after 
  substituting the variables with individual object names) is VALID and 
  could be inserted into the KG; but rather than do that, we simply act 
  on the knowledge that the RDF triple represented by the instantiated 
  'head' of the rule is VALID
'''

#%%

from abc import ABC, abstractmethod
from enum import Enum
import vrd_utils5 as vrdu5

#%%

class LogicMode(Enum):
    CLASSICAL = 1
    FUZZY_PRODUCT = 2

logicMode = LogicMode.CLASSICAL

#%%

def determineRuleOutcome(goalOutcomes):
    '''
    Determine the degree to which a rule has been satisfied based on the
    degrees to which the goals of the rule have been satisfied. Make the
    determination using the mode of logic that's currently active for
    the Rule Engine.
    
    Args
      goalOutcomes - list of numbers representing the outcomes of the
                     goals in the body of a particular rule
    
    Return
      ruleSatisfied - boolean flag indicating whether or not to regard
                      the rule as having been satisfied or not 
      ruleSatLevel - a number in the interval [0,1] representing the degree
                     to which the rule has been assessed to have been 
                     satisfied
    
    LogicMode.CLASSICAL
    - perform standard, classical logic conjunction on the outcomes of the
      goals of the rule; that is, for a rule to be satisfied, all of the
      goals in the body of the rule must have been satisfied
    
    LogicMode.FUZZY_PRODUCT
    - perform 'bold conjunction' on the outcomes of the goals of the rule,
      as defined in Goguen's system of 'product' fuzzy logic; here, if
      u and v are degrees of truth in [0,1], then 'u AND v = uv'; so
      'u AND v AND w = uvw'; etc.
    '''
    
    print(f'goal outcomes: {goalOutcomes}')
    ruleSatisfied = True
    ruleSatLevel = 1
    if logicMode == LogicMode.CLASSICAL:
        for satLevel in goalOutcomes:
            if satLevel != 1:
                ruleSatisfied = False
                ruleSatLevel = 0
                break
    elif logicMode == LogicMode.FUZZY_PRODUCT:
        product = 1
        for satLevel in goalOutcomes:
            product *= satLevel
        if product < 0.3:  # treat low satisfaction levels as 'not satisfied'
            ruleSatisfied = False
            product = 0
        ruleSatLevel = product
    else:
        raise ValueError('conjunctionMode not recognised')
    
    return ruleSatisfied, ruleSatLevel
        

#%%

class BaseRule(ABC):
 
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def shouldRun(self):
        pass

    def evaluate(self, data):       
        goalOutcomes = []
        for goal in self.goals:
            goalOutcomes.append(goal(data))        
        ruleSatisfied, ruleSatLevel = determineRuleOutcome(goalOutcomes)        
        return ruleSatisfied, self.head_predicate, self.ruleId, ruleSatLevel


#%%

# TODO: change PersonalWearableItem to WearableThing in ontology and here
# TODO: this rule is a bit too general and would benefit from refinement
#       to make it more specific; for example, if a person has a hat or
#       or holds a hat, it might well satisfy the current goals 
#       without being on the person's head, in which case saying 'wear'
#       would be inappropriate, but saying 'has' or 'hold' might be 
#       appropriate;
#       see example: 351668053_8d219d6b95_b.jpg (person, hold, hat)
#       to avoid predicting 'wear' as plausible, we'd need to check and find
#       that the position of the hat was not 'on top of' the person but
#       somewhere else; so we need geometric functions to determine the 
#       position of an included object relative to the including object.
#       BUT: the frequency of (person, hold, hat) may be so small that
#       it's not worth bothering trying to catch such exceptions; so check
#       the frequency before refining the rule to catch such exceptions

class WearRule1(BaseRule):
    '''
    wear(X,Y) :- 
        Person(X),
        PersonalWearableItem(Y),
        ¬ Skis(Y),
        ¬ Snowboard(Y),
        ir(Y,X) ~ 1.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'wear'
        self.ruleId = 'rule1'
        self.goals = [self.goal1, self.goal2, self.goal3,
                      self.goal4, self.goal5]

    def goal1(self, data):
        '''
        Person(X) : the subject must be a person
        '''
        subj = data['subject']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome

    def goal2(self, data):
        '''
        PersonalWearableItem(Y) : the object must be wearable item
        '''
        subj = data['object']['id']
        obj = 'PersonalWearableItem'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome

    def goal3(self, data):
        '''
        ¬ Skis(Y) : the object wearable item must not be Skis
        '''
        subj = data['object']['id']
        obj = 'Skis'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if not existsInKG else 0
        return outcome

    def goal4(self, data):
        '''
        ¬ Snowboard(Y) : the object wearable item must not be a Snowboard
        '''
        subj = data['object']['id']
        obj = 'Snowboard'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if not existsInKG else 0
        return outcome
   
    def goal5(self, data):
        '''
        ir(Y,X) ~ 1 : The inclusion ratio of the object bbox relative to
                      the subject bbox must be at or very near 1. That is,
                      the object bbox must be fully enclosed (or very nearly
                      so) within the subject bbox (as would be the case if
                      the person was wearing the item in question).
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        _, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1, b2)
        if logicMode == LogicMode.CLASSICAL:
            outcome = 1 if ir_b2b1 > 0.90 else 0
        else:  # FUZZY logic of some sort
            # an inclusion ratio always lies in the interval [0,1], so we
            # can use the value of the inclusion ratio as a 'degree of truth'
            # for the condition that the subject bbox is enclosed within
            # the object bbox; the closer to 1, the more the subject
            # bbox is enclosed within the object bbox and the more the
            # goal is satisfied; the closer to 0, the less it is enclosed
            # and hence the less the goal is satisfied
            outcome = round(ir_b2b1, 2)   
        return outcome

    def shouldRun(self, data):
        runRule = False
        if data['subject']['class'] in [0]:  # subject is a Person
            runRule = True 
        return runRule

#%%

class WearRule2(BaseRule):
    '''
    wear(X,Y) :- 
        Person(X),
        Skis(Y),
        ir(Y,X) > .1.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'wear'
        self.ruleId = 'rule2'
        self.goals = [self.goal1, self.goal2, self.goal3]

    def goal1(self, data):
        '''
        Person(X) : the subject must be a person
        '''
        subj = data['subject']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome

    def goal2(self, data):
        '''
        Skis(Y) : the object wearable item must be Skis
        '''
        subj = data['object']['id']
        obj = 'Skis'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
    
    def goal3(self, data):
        '''
        ir(Y,X) > .1 : The inclusion ratio of the object bbox relative to
                       the subject bbox must be at or very near 1. That is,
                       the object bbox must be fully enclosed (or very nearly
                       so) within the subject bbox (as would be the case if
                       the person was wearing the item in question).
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        _, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1, b2)
        outcome = 1 if ir_b2b1 > 0.1 else 0
        return outcome

    def shouldRun(self, data):
        runRule = False
        if data['subject']['class'] in [0]:  # subject is a Person
            runRule = True 
        return runRule 

#%%

class WearRule3(BaseRule):
    '''
    wear(X,Y) :- 
        Person(X),
        Snowboard(Y),
        ir(Y,X) > .1.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'wear'
        self.ruleId = 'rule3'
        self.goals = [self.goal1, self.goal2, self.goal3]

    def goal1(self, data):
        '''
        Person(X) : the subject must be a person
        '''
        subj = data['subject']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome

    def goal2(self, data):
        '''
        Snowboard(Y) : the object wearable item must be Snowboard
        '''
        subj = data['object']['id']
        obj = 'Snowboard'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
    
    def goal3(self, data):
        '''
        ir(Y,X) > .1 : The inclusion ratio of the object bbox relative to
                       the subject bbox must be at or very near 1. That is,
                       the object bbox must be fully enclosed (or very nearly
                       so) within the subject bbox (as would be the case if
                       the person was wearing the item in question).
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        _, ir_b2b1 = vrdu5.bb_inclusion_ratios(b1, b2)
        outcome = 1 if ir_b2b1 > 0.1 else 0
        return outcome

    def shouldRun(self, data):
        runRule = False
        if data['subject']['class'] in [0]:  # subject is a Person
            runRule = True 
        return runRule 


#%%

class OnRuleWear1(BaseRule):
    '''
    on(X,Y) :- 
        PersonalWearableItem(X),
        ¬ Skis(X),
        ¬ Snowboard(X),
        Person(Y),
        ir(X,Y) ~ 1.
    
    This rule describes cases where it's valid to use predicate 'on' in the 
    sense of the inverse of the predicate (verb) 'wear'. For example:
    <jacket, on, person>, <shorts, on, person>, <sunglasses, on, person>.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'on'
        self.ruleId = 'wear1'
        self.goals = [self.goal1, self.goal2, self.goal3, 
                      self.goal4, self.goal5]

    def goal1(self, data):
        '''
        PersonalWearableItem(X) : the subject must be a wearable item
        '''
        subj = data['subject']['id']
        obj = 'PersonalWearableItem'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome

    def goal2(self, data):
        '''
        ¬ Skis(X) : the subject wearable item must not be Skis
        '''
        subj = data['subject']['id']
        obj = 'Skis'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if not existsInKG else 0
        return outcome

    def goal3(self, data):
        '''
        ¬ Snowboard(X) : the subject wearable item must not be Snowboard
        '''
        subj = data['subject']['id']
        obj = 'Snowboard'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if not existsInKG else 0
        return outcome    

    def goal4(self, data):
        '''
        Person(Y) : the object must be a person
        '''
        subj = data['object']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
    
    def goal5(self, data):
        '''
        ir(X,Y) ~ 1 : The inclusion ratio of the subject bbox relative to
                      the object bbox needs to be at or very near 1. That is, 
                      we want the subject bbox to be fully enclosed (or very
                      nearly so) within the object bbox.  That is, the
                      wearable item must have a spatial association with the
                      person in question that would make it plausible to 
                      say the person is wearing the item, in which case it
                      becomes plausible to say the item is 'on' the person.
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        ir_b1b2, _ = vrdu5.bb_inclusion_ratios(b1, b2)
        if logicMode == LogicMode.CLASSICAL:
            outcome = 1 if ir_b1b2 > 0.90 else 0
        else:  # FUZZY logic of some sort
            # an inclusion ratio always lies in the interval [0,1], so we
            # can use the value of the inclusion ratio as a 'degree of truth'
            # for the condition that the subject bbox is enclosed within
            # the object bbox; the closer to 1, the more the subject
            # bbox is enclosed within the object bbox and the more the
            # goal is satisfied; the closer to 0, the less it is enclosed
            # and hence the less the goal is satisfied
            outcome = round(ir_b1b2, 2)
        return outcome
      
    def shouldRun(self, data):
        runRule = False
        if data['object']['class'] in [0]: # object is a Person
            runRule = True 
        return runRule 

#%%

class OnRuleWear2(BaseRule):
    '''
    on(X,Y) :- 
        Skis(X),
        Person(Y),
        ir(X,Y) > .1.
    
    This rule describes cases where it's valid to use predicate 'on' in the 
    sense of the inverse of the predicate (verb) 'wear'. For example:
    <skis, on, person>.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'on'
        self.ruleId = 'wear2'
        self.goals = [self.goal1, self.goal2, self.goal3]

    def goal1(self, data):
        '''
        Skis(X) : the subject needs to be Skis
        '''
        subj = data['subject']['id']
        obj = 'Skis'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
   
    def goal2(self, data):
        '''
        Person(Y) : the object must be a person
        '''
        subj = data['object']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
    
    def goal3(self, data):
        '''
        ir(X,Y) > .1 : the inclusion ratio of the subject bbox relative to
                       the object bbox needs to be greater than 0 (ie the
                       two bboxes must intersect to some degree), but we 
                       don't require that the subject bbox be (nearly) fully
                       enclosed within the object bbox because the bboxes for
                       skis and snowboards may often be well outside the bbox
                       of the person wearing them
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        ir_b1b2, _ = vrdu5.bb_inclusion_ratios(b1, b2)
        outcome = 1 if ir_b1b2 > 0.1 else 0
        return outcome
    
    def shouldRun(self, data):
        runRule = False
        if data['object']['class'] in [0]: # object is a Person
            runRule = True 
        return runRule 

#%%

class OnRuleWear3(BaseRule):
    '''
    on(X,Y) :- 
        Snowboard(X),
        Person(Y),
        ir(X,Y) > .1.
    
    This rule describes cases where it's valid to use predicate 'on' in the 
    sense of the inverse of the predicate (verb) 'wear'. For example:
    <snowboard, on, person>.
    '''

    def __init__(self, kg):
        self.kg = kg
        self.head_predicate = 'on'
        self.ruleId = 'wear3'
        self.goals = [self.goal1, self.goal2, self.goal3]

    def goal1(self, data):
        '''
        Snowboard(X) : the subject needs to be Snowboard
        '''
        subj = data['subject']['id']
        obj = 'Snowboard'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
   
    def goal2(self, data):
        '''
        Person(Y) : the object must be a person
        '''
        subj = data['object']['id']
        obj = 'Person'
        existsInKG = self.kg.checkTripleInKG(subj, obj)
        outcome = 1 if existsInKG else 0
        return outcome
    
    def goal3(self, data):
        '''
        ir(X,Y) > .1 : the inclusion ratio of the subject bbox relative to
                       the object bbox needs to be greater than 0 (ie the
                       two bboxes must intersect to some degree), but we 
                       don't require that the subject bbox be (nearly) fully
                       enclosed within the object bbox because the bboxes for
                       skis and snowboards may often be well outside the bbox
                       of the person wearing them
        '''
        b1 = data['subject']['bbox']
        b2 = data['object']['bbox']
        ir_b1b2, _ = vrdu5.bb_inclusion_ratios(b1, b2)
        outcome = 1 if ir_b1b2 > 0.1 else 0
        return outcome
    
    def shouldRun(self, data):
        runRule = False
        if data['object']['class'] in [0]: # object is a Person
            runRule = True 
        return runRule 
        
#%%

class RuleEngine():
    
    def __init__(self, vrdKG):
        
        # instantiate the knowledge base of rules, passing each rule a 
        # reference to a (wrapped) knowledge graph
        self.rules = []
        self.rules.append(WearRule1(vrdKG))
        self.rules.append(WearRule2(vrdKG))
        self.rules.append(WearRule3(vrdKG))
        self.rules.append(OnRuleWear1(vrdKG))
        self.rules.append(OnRuleWear2(vrdKG))
        self.rules.append(OnRuleWear3(vrdKG))
    
    def execute(self, data):
        
        satisfiedRules = {}
        
        for rule in self.rules:
            if rule.shouldRun(data):
                #print(f'evaluating rule: {rule.head_predicate} {rule.ruleId}')
                result = rule.evaluate(data)
                #print(f'rule result: {result}')
                satisfied, predicate, ruleId, satLevel = result
                if satisfied:
                    if not predicate in satisfiedRules:
                        satisfiedRules[predicate] = {}
                    satisfiedRules[predicate][ruleId] = satLevel
        
        return satisfiedRules




















