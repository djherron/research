# Rule Ideas for Rule Engines {ignore=true}

**Table of Contents**
[TOC]

# image analysis for discovering high-impact rules

analyse the frequency distribution of VR types; use this to identify the rules that will give us the 'biggest bang for our buck'

identify all unqiue visual relationship types, (s, p, o), and count them and rank them in descending order of frequency
* per training set
* per test set
ensure we have rules covering the most frequently occurring vr types, and don't waste time conceiving and implementing rules for vr types that occur infrequently and, hence, will have little impact on predictive performance

use image analysis tools already developed to find sets of vrs that are good candidates for rules
* find all 'subject' classes for given (predicate, object)
* find all 'object' classes for given (subject, predicate)
* etc


# geometric rules feature functions required

function(s) to calculate position of subject object relative to object object
* needed to determine if subject object is 'on top of' (eg hat on head as opposed to hat not on head), 'at bottom of', 'at side of', 'above', 'below', 'beside', 'on the left of', 'on the right of', etc, so we can build rules to catch the following:
* (X, above, Y)
* (X, below, Y)
* (X, behind, Y)
* (X, in front of, Y)
* (X, beside, Y), (X, next to, Y), (X, adjacent to, Y)
* (X, near, Y), (X, by, Y)
* etc.


# changes to existing rules

rules: (person, wear, Y), (X, on, person)
* change the goals that currently refer to the specific class Person to refer to the union class (more general class) 'WearCapableThing'
* current classes in the union: Person, Teddy Bear
* this way, we can augment the member classes of WearCapableThing without the rule having to change; plus, a goal that depends on this union class now must rely on the reasoning capabilities of the KG in order to be satisfied
* this is an example of general rule 'design pattern'; ie never use a specific class; always introduce a more general equivalent/union class and build the rule by referring to that more general class


# ideas for specific rules

## under/below/beneath and over/above

(street, under, Y), (street, below, Y), (street, beneath, Y)
* if street is the subject, then 'under', 'below' and 'beneath' are plausible predicates for any object object Y
* geometric features aren't necessary to make this determination

(sky, over, X), (sky, above, X)
* if sky is the subject in an object pair, it is plausible (and common) that it be 'over', 'above' any object object X
* geometric features aren't necessary to make this determination

(X, below, sky), (X, under, sky), (X, beneath, sky)
* if object is sky, then predicates 'below', 'under', 'beneath' are plausible for any subject X
* geometric features aren't necessary to make this determination

(X, over, street), (X, above, street)
* if object is street, then predicates 'over' and 'above' are plausible for any subject X
* geometric features aren't necessary to make this determination

(person, under, umbrella)
* need geometric function for relation 'under', 'below', 'beneath'
* or could just treat it like 'has' (ie relying on an inclusion ratio, without any sense of relative position)

## 'on' related

(X, on, street)
* if object is street and inclusion ratio (intersection area) is appropriate, then predicate 'on' is plausible for any subject X

(X, park on, street)
* if object is street and subject X is a 'drivable, wheeled vehicle' (car, van, truck, bus), and inclusion ratio is appropriate, then predicate 'park on' is plausible

(X, drive on, street)
* if object is street and subject X is a 'drivable, wheeled vehicle' (car, van, truck, bus), and inclusion ratio is appropriate, then predicate 'drive on' is plausible

(person, stand on, street)
* if subject is person and object is street and inclusion ratio is appropriate, then 'stand on' is plausible
* this could be extended to subjects that are 'standable things' (eg person, horse, giraffe, elephant)

(person, on, grass)
* as per street

(person, on, X)
* X is an 'OnableSurface' (street, grass); sometimes the inclusion ratio can be zero due to occlusions; eg 3287856690_a81bec1e0f_b.jpg (person, on, street)
* X is an 'OnableVehicle' (truck, bus, train, boat, bike, motorcycle, ...) (not 'car' or 'airplane'; because while it's possible to have an image with a person 'on' a car, it's very uncommon (in VRD dataset) to have or say <person, on, car> or <person, on, airplane)
* X is an 'OnableSportingGood' (skis, snowboard, skateboard, surfboard)
* X is an 'OnableAnimal' (horse, elephant); usually the inclusion ratio is above 0, but there can be things occluding parts of the person and/or animal, so sometimes the inclusion ratio can be 0; eg 4357819482_97dd8280b1_b.jpg (person, on, horse);

(person, stand on, grass)
* as per street

(person, on, X)
* if subject is 'person' and object is a 'sittable thing' (eg chair, sofa, bench, etc), and inclusion ratio is appropriate, then predicates 'on' is plausible

(person, sit on, X)
* if subject is 'person' and object is a 'sittable thing' (eg chair, sofa, bench, etc), and inclusion ratio is appropriate, then predicates 'sit on' is plausible

(X, on, table) or (X, on, desk) or (X, on, counter) or (X, on, shelf)
* if object is (table or desk) and subject X is a 'table-top thing' (eg plate, bowl, drinking glass, bottle, cup, can, phone, camera, laptop, keyboard, mouse, monitor, speaker, vase, lamp, pot, etc) and inclusion ratio is appropriate, then 'on' is a plausible predicate
* use (or create) a class that unions 'table', 'desk', counter, shelf into one class; eg FlatSurfaceFurniture class already exists in vrd_dh_custom ontology

(pillow, on, Y)
* if subject is pillow and Y is (SeatingFurniture or SleepingFurniture) and inclusion ratio is appropriate, then predicate 'on' is plausible
* create a class that unions SeatingFurniture and SleepingFurniture into one; eg pillowable-furniture  

## on, cover, has

(trees, on, mountain), (trees, cover, mountain), (mountain, has, trees)
* if inclusion ratio is appropriate, then predicate is plausible


## has, in, with

(person, has, X)
* if person is the subject and X is a 'WearableThing', and inclusion ratio is appropriate, then 'has' is a plausible predicate
* NOTE: the inclusion ratio is not always near 1; eg in image 4347436774_5f447b1fc0_b.jpg, (person, has, hat) had an inclusion ratio of just 0.094  because the bbox for the person only included a small portion of the hat (i since changed the bbox for person to include the hat)
* here 'has' may represent a synonym for 'wear'
* but it can also be used as a synonym for hold or carry or eat
* if X is not a WearableThing but something like a 'laptop', you can find images where the inclusion ratio is 0.0; 1581060036_b82eb3cddd_b.jpg has (person, has laptop) with an ir of 0.0 because there's a chair occluding part of the person

(person, in, X)
* if person is the subject and X is a 'WearableThing', and inclusion ratio is appropriate, then 'in' is a plausible predicate
* here 'in' is a synonym for 'wear'
* X is also often 'street', 'grass' ('InableSurface')
  - inclusion ratio of 'street' within 'person' will be tiny (usually)
  - inclusion ratio of 'person' within 'street' may be full or partial
* X is also often a vehicle (car, truck, bus, van, boat, train) 'InableVehicle'
  - here the inclusion ratio of X within 'person' may be tiny
  - the inclusion ratio of person within X should be at or near 1

(person, in, chair)
* if subject is 'person' and object is 'chair' and inclusion ratio is appropriate, then predicates 'in' is plausible
* we don't tend to say (person, in, sofa) or (person, in, bench), but (person, in, chair) is common

(person, with, X)
* if person is the subject and X is a 'WearableThing', and inclusion ratio is appropriate, then 'in' is a plausible predicate; here 'in' is a synonym for 'wear'
* if X is 'person', sometimes (more often) the subject is holding X (a child), and the inclusion ratio is near 1; but sometimes the inclusion ratio is 0 and we just have 2 people near each other in a room; this latter case is akin to a scenario where the predicate 'near' or 'beside' would apply; ie here 'with' is being used as a synonym for 'near'. 'beside', 'next to'; eg 4930159425_18830d74cf_b.jpg, 9285769359_9e015a3865_b.jpg
* sometimes X is a HoldableThing (like a camera, phone, bag, umbrella, etc); here the inclusion ratio is usually strong (near 1) since here 'with' is used as a synonym for 'has', 'hold' or 'carry'; but sometimes X is just close to the person, and the inclusion ratio is zero (0); eg 9046906485_f3b1798e23_b.jpg (person, with, bottle)
* sometimes X is a 'bike' and the person may be riding it or holding it or walking with it next to them
* similarly, we might have (person, with, kite) where the inclusion ratio is 0 and the two bboxes are far apart


## hold, has

(person, hold, X)
* if subject is 'person' and object X is a 'holdable thing' (eg phone, camera, bottle, drinking glass, bag, umbrella, pizza, etc) and inclusion ratio (or intersection area) is appropriate, then 'hold' is a plausible predicate
* (person, hold, person) is not uncommon, where an adult is holding a child; so 'person' can also be a 'holdable thing'; here the inclusion ratio would be near 1 usually, like a piece of clothing (WearableThing)

(hand, hold, X)
* if subject is hand and X is a 'holdable thing' (eg phone, camera, bottle, drinking glass, bag, pizza, etc) and inclusion ratio near 1, then 'hold' is a plausible predicate

(person, has, X)
* if subject is 'person' and object X is a 'holdable thing' (eg phone, camera, bottle, drinking glass, bag, umbrella, etc) and inclusion ratio (or intersection area) is appropriate, then 'has' is a plausible predicate
* (person, has, person) sometimes occurs; see (person, hold, person)
* 7704320214_8ee60db9c4_b.jpg has (person, has, cup) with ir 0.036; so ir can be very near zero
* 6181281417_440d85955b_b.jpg (person, has, skateboard) with ir 0.082
* 6157309791_c66e93a6d1_b.jpg (person, has, bag) with ir 0.061

(person, has, umbrella), (person, hold, umbrella)
* inclusion ratio can be minimal, esp if it's held high and the stem isn't part of the bbox, just the fabric
* 5868606716_b168d41f59_b.jpg (person, has, umbrella); ir is just 0.076
* 5359614021_19f2cc7039_b.jpg (person, has, umbrella); ir is 0.0 because stem isn't part of the umbrella bbox
* 7323344210_9c75cf09c7_b.jpg (person, has, umbrella); ir 0.097
* 7890896924_a553b5deca_b.jpg (person, has, umbrella); ir 0.155
* 8476650893_56d25c7356_b.jpg (person, has, umbrella); ir 0.054
* 6259476716_eb5eec2ab1_b.jpg (person, has, umbrella); ir 0.036
* 8449584027_4964ce99ac_b.jpg (person, has, umbrella); ir 0.023

(person, has, suitcase)
* this is like (person, has, umbrella) in that the inclusion ratio can often be very small or even 0, like when the person is pulling the wheeled suitcase and the extended hand isn't fully included in the bbox of the suitcase (or even if it is)
* 1414821862_846f743d67_b.jpg (person, has, suitcase); ir 0.067

(person, has, hand)
* if subject is person and object is hand and inclusion ratio near 1, then 'has' is a plausible predicate


## attached to, has

we can't assert the OWL rule that 'attached to' and 'has' are inverses because this is too general; sometimes they do work as inverses, but not always; but there is a kind of 'conditional inverse' relationship between 'attached to' and 'has'; if we determine that (A, attached to, B), then we can assert (B, has, A); what we really need here, in OWL, is a construct such as 'subInverseOf' (a property that's a subproperty of another but only as an inverse); then we could assert that (attached to, subInverseOf, has)
* Datalog rule: (B, has, A) :- (A, attached to, B)
* the issue here is that 'has' is general enough (in natural language terms) to convery the notion of 'possession', the notion of 'wearing' and the notion of 'part of'; if we're sure of the 'part of' notion, then we can use 'has' as an inverse in the more general setting; but if we're only sure of 'has' (in the general setting), we can't assert any of the 'conditional' inverse relationships because we don't know which one applies ('possession' --> no inverse, 'wearing' --> 'wear', 'part of' --> 'attached to')
* if this notion of 'subInverseOf' satisfied the conditions for decidability within a Description logic (like OWL), then it would be a useful construct to introduce in a future version of OWL;

(X, has, wheel)
* if object is wheel and X is a 'wheeled vehicle thing' (car, van, truck, bus, bike, motorcycle, etc) and inclusion ratio is appropriate, then 'has' is a plausible predicate

(wheel, attached to, Y)
* if subject is wheel and Y is a 'wheeled vehicle thing' and inclusion ratio is appropriate, then 'attached to' is a plausible predicate


## ride

(person, ride, Y)
* if subject is person and object Y is a 'rideable thing' (eg horse, elephant, surfboard, skateboard) and inclusion ratio is appropriate (appropriate will vary be object type) and the position of the person is 'on top of' the object, then 'ride' is a plausible predicate
* create a union class 'ridable thing' (horse, elephant, surfboard, skateboard, etc)
* for position of person re object Y we could use 'cosine of angle between centroids', which, in this case, should be near 0  (ie the centroid of person bbox should be at a 90 degree angle relative to the centroid of the horse bbox)
* or we need to invent another way of detecting 'on top of'
* the inclusion ratio may need to vary with the object type, so it may be we need separate rules for horse, elephant and (surfboard/skateboard)
* check (person, ride, skis); uncommon but it occurs; do we want to keep it or eliminate it? if keep, don't include skis as a 'ridableThing' since it's so uncommon to use 'ride' in this context
* check (person, ride, snowboard); same issue, but here 'ride' is a bit more acceptable perhaps


## at, beside, next to

(person, at, table), (person, at, counter), (person, at, desk)
* if subject is person and object is table and inclusion ratio is appropriate (some intersection but not too much), then 'at' is a plausible predicate
* create new class

(person, beside, table)
* if subject is person and object is table and inclusion ratio is appropriate (some intersection but not too much), then 'beside' is a plausible predicate

(person, next to, table)
* if subject is person and object is table and inclusion ratio is appropriate (some intersection but not too much), then 'next to' is a plausible predicate

(person, next to, person)
* if subject is person and object is person and IoU (or inclusion ratio) is appropriate, then 'next to' is a plausible predicate


## behind

(mountain, behind, Y)
* usually, if there's a mountain, it is said to be 'behind' other things (person, vehicle, etc)


## wear, has, in

(person, wear, Y), (person, has, Y)

TODO: change 'person' to something more general: WearCapableThing
* replace the specific class Person with the more general class WearCapableThing; this way, the ontology can be augmented with additional 'wear-capable' things without the rule in the Rule Engine having to change; plus, the goal that checks if X is a WearCapableThing will now have to rely on the reasoning capabilities of the KG whereas before, with the specific class Person, it had to rely only the KG's capabilities as a data store (to store the inserted triple <person123, rdf:type, Person)
* something that is capable of being said to 'wear' something
* Person, Teddy Bear

* the inclusion ratio need not be near 1 for 'wear' to be plausible, even for things that are not skis or snowboards; it depends on the person's posture, etc.
* 8726808065_b2913ec5aa_b.jpg has a valid (person, wear, pants) with ir 0.484
* 7890896924_a553b5deca_b.jpg has (person, wear, jacket) with ir 0.473 (it's a rain cape that's blowing wide in the wind)
* 6800090271_500d63e565_b.jpg has (person, wear, hat) with ir 0.326 (the bbox for the person doesn't encompass the whole of the hat, just a portion of it)
* when Y is skis, I've seen incl ratios as low as 0.235, 0.211, 0.202 and even 0.055 (5523729938_b06e69ce92_b.jpg, although the bboxes for person and skis have since been improved which will have altered the inclusion ratio, but it will still be small)


(person, in, Y)

notes re (person, in, Y)
* 'in' is only sometimes as synonym for 'wear', so we can't define 'wear' as a subPropertyOf 'in' in the class hierarchy of the ontology; this is where rules in a rule engine can help, because a specialised rule can be constructed to catch the subset of cases where 'in' is used as a synonym for 'wear'
* the rule for (person, in, Y) can be very close to the rule for (person, wear, Y), but the class of Y for 'in' needs to be more specialised than the class of Y for 'wear'; we need a 'union class' named 'InableThing' --- something that a person can be said to be 'in'; an 'InableThing' must be Clothing


## carry

(person, carry, Y)
* Y needs to be a 'CarryableThing'
* analysis_3.py showed that, in training set, Y is any of the following:
bag
bottle
snowboard
skis
umbrella
surfboard
skateboard
suitcase
person
plate
cart
pillow
camera
cup
jacket
* TODO: investigate suitcase, plate, cart, pillow, camera, cup, jacket
* TODO: define 'CarryableThing' in ontology
* TODO: define rule in rule engine





# Rule Hierarchy

KEY POINT: the rules can form a hierarchy that mirrors the property hierarchy
* if a rule for a property is satisfied, then a rule for any parent property of that property is also satisfied !
* if a rule for a property is satisfied, then a rule for any equivalent property of that property is also satisfied !
* this can be used to simplify the definition of many rules, especially in an RDFox implementation

in an OWL ontology we have the concepts 'class hierarchy' and 'property hierarchy'; in the context of a Rule Engine, it is reasonable to posit and explore the viability of the concept 'rule hierarchy'

'wear' vs 'has'
* in the class hierarchy of the ontology, 'wear' is a subPropertyOf 'has', so if (person, wear, Y) is an asserted triple, then (person, has, Y) is an inferrable triple
* looking at the relationship in the reverse direction, we can say that 'wear' is a specialsed version of 'has'; or, more simply, 'wear' is a specialisation of the more general property 'has'
* similarly, in constructing rules for our rule engine, we observe that an appropriate rule for inferring the plausibility of (person, wear, Y) turns out to be a specialised version (or specialisation) of the rule for inferring the plausibility of (person, has, Y)
* more specifically, in the former rule, one goal insists that Y be a 'WearableThing', whereas in the latter rule, one goal insists that Y be a 'HasableThing', and it turns out that the classes in the OWL union class 'WearableThing' are a subset of the classes in the OWL union class 'HasableThing'
* so, conceptually, the rule for (WearCapableThing, wear, WearableThing) can be conceived as being a 'subRuleOf' the rule for (HasCapableThing, has, HasableThing)
* notice that this example of the concept of 'subRuleOf' mirrors the relationship between the properties 'wear' and 'has' in the ontology's property hierarchy; 'wear' is modelled as a subPropertyOf 'has', that is, 'wear' is a specialisation of 'has' that applies to a more restricted set of object types (classes)

(X, on, person)

r1 (Y, on, X) :- (X, wear, Y)

ie this is a case where 'wear' could be (perhaps should be) defined as a subPropertyOf 'on';  so if rule (X, wear, Y) is satisfied, then we know rule (Y, on, X) is satisfied; but things don't work in the opposite direction because 'on' is too general a property (predicate); it has too many 'use cases'


IDEAS FOR IMPLEMENTING THE RULE HIERARCHY CONCEPT

One way to implement the concept of 'subRuleOf' and Rule Hierarchy is to extend the sophistication of the Rule Engine by introducing support for a limited amount of rule chaining/recursion. For example, conceptually we want to define the following:

r1 (X, has, Y) :- (X, wear, Y)

r4 (X, wear, Y) :- (X, type, WearCapableThing),
                   (Y, type, WearableThing),
                  ¬(Y, type, WearableSportingGood),
                   ir(Y,X) > 0.5

r5 (X, wear, Y) :- (X, type, WearCapableThing),
                   (Y, type, WearableSportingGood),
                   ir(Y,X) > 0.1

So, rule r1 is satisfied if rule r4 or rule r5 is satisfied

One way to support this type of rule chaining is to store the rule objects (instances of rule classes) in a Python Dictionary that we can use as a look-up table. If we detect that the body of rule is a subRule (ie the head of another rule), then we lookup that rule in the Dictionary and call it.

By supporting this concept of Rule Hierarchy and a 'subRuleOf' relationship, we need to code fewer rules. We can simply call the subRule instead.


# Other Ideas

if a rule is satisfied (ie the predicate in the head is deemed plausible), then check the KG for 'equivalent' properties and automatically deem those predicates to be plausible as well; (ie don't bother implementing separate rules for the equivalent properties, rely solely on the 'equivalence' with the other predicates as defined in the ontology of the KG)

same idea but with subPropertyOf relationships; if a rules is satisfied and the predicate in the head is a subPropertyOf another property in the KG, then automatically deem the super-property to be plausible; executing a separate rule for the super-property will be redundant at that point


OTHER IDEA: score = relative freq of VR amongst gt VRs * confidence of predicted VR

let the empirical relative frequencies of the VRs amongst the training set annotated VRs have influence; so rather than making decisions about which VRs to predict based on the NN confidence scores alone, temper these scores with the empirical relative frequency info from the training set
 * use a 'product', A * B, to combine the two components of info
 * base prediction decisions upon the resulting score
 * the most likely VRs (predicates) will be those that have both a high relative freq in training set and a high NN prediction score
 * if both A and B are large, then A*B will be large
 * if one of A or B is small, then A*B will be small
 * if both A and B are small, then A*B will be very small
 * if A (relative freq) is small but B is very large, then A*B may still be large enough to get selected as a predicted VR; this would allow for zero-shot, few-shot predictions to get through
   - maybe introduce another parameter to allow the effect/influence of the relative frequency component, A, to be controlled/moderated
   - maybe during training we want A to have full influence, but during inference we want to dampen its influence to better allow zero-shot/few-shot predictions to get selected for submission, where the NN is confident (and B is large) and we want to go with that confidence rather than reject the prediction because relative freq in the training set is low
   - $\lambda AB$, where $\lambda$ can be used to control the influence of $A$; for example, if we set $\lambda = \frac{1}{A}$ then $\lambda A = 1$ and the value $B$ will be controlling the score all by itself
   - during training, set $\lambda = 1$ to $A$ can influence things; during inference, set $\lambda = \frac{1}{A}$ to nullify the influence of $A$

maybe combine a 3rd component, C, the plausibility result from the Rule Engine
* so: score = A * B * C
or $score = \lambda A * B * C$
