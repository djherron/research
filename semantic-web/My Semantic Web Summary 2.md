# My Semantic Web Summary 2 {ignore=true}

This is my summary of the Semantic Web technology stack (RDF, RDFS, OWL, etc.) and of topics relating to ontology modelling/engineering using the technology stack.  The primary source for this summary is `Semantic Web for the Working Ontologist`, 3ed. But info and insights from other sources have been (and continue to be) introduced as well.

The best way to view and use this document is to render it within an editor that supports Markdown previewing. We prefer editors Atom and MS Visual Studio Code, both of which allow installation of a useful support package called `Markdown Preview Enhanced`. Amongst other things, this Markdown rendering package interprets the `[TOC]` (Table of Contents) command and builds a dynamic Table of Contents with automatic internal hyperlinks to each Heading, SubHeading, etc., This is extremely useful for navigating large documents like this one: useful for viewing the contents/structure and for jumping to sections of interest.  The `Markdown Preview Enhanced` package also provides a convenient `back to top` hyperlink icon for jumping back to the top of the document.


**Table of Contents**
[TOC]

# The Semantic Web Assumptions

Unlike other modelling systems such as object-oriented (OO) systems, Semantic Web models are intended to work in the context of 3 assumptions that we call the Semantic Web assumptions.

## The AAA (Anyone can say Anything about Any topic) assumption

The *Anyone can say Anything about Any topic* (AAA) assumption:
* on the Web, we are all content producers; we can say anything we like (on web sites, web pages, blog posts, etc.); anyone can write a web page saying whatever they please and publish it to the Web infractructure
* for the Semantic Web, this means that our architecture has to allow any individual to express a piece of data about some entity in a way that can be combined with data from other sources
* it also means that the Web is like a data wilderness --- full of treasure but overgrown and tangled; there is no single database administrator who rules with an iron first and enforces total standardisation; the Web has no gate keeper; a distributed web of data is an organic system, with contributions coming from all sources; this freedom of expression on the Web is what allowed it to take off as a bottom-up, grassroots phenomenon
* the AAA slogan enables the network effect that made the rapid growth of the Web possible
* implications of the AAA slogan:
  - we have to be prepared to cope with a wide range of variation in the information on the Web
  - sources of variation: people may hold contrary opinions (ie disagree) on a topic; someone might want to intentionally deceive; someone might simply be mistaken; some information may be out of date; from a technical perspective, there is no way to distinguish the source(s) of a variance; information on the Web will disagree from time to time; this is a permanent condition at the heart of the very nature of the Web (variations and disagreement)
  - different Web authors will select different URIs for the same real-world resource

## The OWA (Open World Assumption)

The *Open World* assumption (OWA):
* we must consider the Web as an *open world*; we must assume at any time that new information may come to light; we may draw no conclusions that rely on assuming that the information currently available at any one point is all the information available
* we only draw conclusions from data that will not be undermined or countermanded by the discovery of new data; this assumption makes sense when we are interpreting data we find 'in the wild', like data on the Web
* many Semantic Web applications can ignore the Open World Assumption; where the openness of the Web becomes an issue is when we want to draw conclusions based on distributed data; for example, just because we may not currently have a certain fact in our knowledge base, we cannot assume that no such fact exists; we simply may not be aware of it or it may currently not exist but could materialise at any point in the future

## The NUNA (Non-unique Naming Assumption)

The *non-unique naming* assumption (NUNA):
* we cannot assume that just because two URIs are distinct, they refer to distinct resources
* it means we cannot tell (assume) that two entities are distinct just by looking at their URIs and noticing that they are different
* so we make no inferences that would preclude the possibility that they refer to the same resource unless and until we are informed otherwise
* we have to assume (until told otherwise) that some Web resource might be referred to using different names by different people
* just because two resource URIs are different, we don't assume they belong to different resources; we remain open to the possibility that the two URIs may refer to the exact same resouce unless and until we are informed otherwise (either by assertion or by inference)

## The CWA (Closed World Assumption)

the *closed world* assumption (CWA)
* in contrast to the Open World Assumption (OWA), most data systems operate under the Closed World Assumption (CWA); that is, if we are missing some data, then that data is simply not available (ie we assume it does not exist)



# RDF

RDF (Resource Description Framework)

## Overview

RDF models data as a graph
* RDF allows us to represent data as a graph
* by itself, RDF simply lets us create a graph structure to represent data
* it's a data model based on a graph model --- a collection of *nodes* and *edges*
* it supports a *directed* graph model, with *labelled* edges
* *nodes* are either: resources, literals or blank nodes
* *edges* are: resources

RDF describes *resources*
* in the Semantic Web we refer to the things in the world as *resources*; a *resource* can be anything that someone might want to talk about
* synonyms for *resource* that are sometimes used are 'entity' or 'thing'
* an RDF resource can be a: individual, class or property
* *resources* are identified by HTTP URIs/IRIs
* a URI/IRI (borrowed from foundational Web technology) is a globally unique identifier; typically this is an HTTP URL

RDF enables a WWW of data
* it addresses one fundamental issue: managing distributed data (on a global scale)
* it provides a model of data that allows information (data) to be distributed across the web and yet be linked/integrated
* by allowing us to represent data as a graph, and by requiring us to use HTTP URIs/IRIs as identifiers of resources, RDF allows us to weave a WWW of *linked data* (a *WWW of data* that exists in parallel to the classic *WWW of documents*)
* global naming via URIs/IRIs leads to global network effects
* merged information from two sources (graphs) is simply the union of the two graphs, forming a larger graph; a node from one graph is merged with a node from another graph if they have the same URI

RDF is more about individuals than about classes and properties
* although an RDF *resource* can be an individual, class or property, RDF, by itself, primarily allows one to describe resources that are *individuals*
* that is, with RDF alone, the triples one uses tend to be `<individiual, property, value>` triples    
* RDFS extends RDF to allow one to 1) define and describe *classes* and 2) to describe *properties* more richly
* so, while an RDF *resource* can be an individual, class or property, when we confine ourselves to the context of RDF alone (ignoring RDFS, OWL, etc.), it's useful to keep in mind that *resource* is primarily about *individuals*
* in other words, with RDF alone, we can't define resources in terms of other resources; for that, we need RDFS and OWL  

Additional stuff
* is the foundation of the semantic web technology stack
* it's a general purpose language for describing structured information
* it's the basic representation format underlying the Semantic Web
* it's a data format that facilitates data exchange (data interchange)


## RDF triples

In RDF, everything is expressed as triples:
`<:subject :predicate :object>`
* an RDF triple is also called a *statement* or *fact*
* subject: the thing (resource) that a statement is about
* predicate: a property of the subject
* object: the value of the predicate (property), which may be a resource or a literal

RDF triples
* as seen above, the elements of an RDF triple can be:
  - resources (identified by URIs/IRIs)
  - literals
  - blank nodes
* a *resource* can be a class, property or individual
* *literals* can be strings, numbers, dates, etc.; literals are just values; no relationships from literals are allowed; literals are used to describe individuals; eg <:John, :hasAge, :29>
* *blank nodes* are like resources without a URI (ie with no web identity); they're used when a resource is unknown or has no (natural) identifier; they are used as a mechanism to link multiple triples when no natural URI exists to do the linking

RDF triple position rules:
* *resources* (identified by  *URIs/IRIs*) may appear in any triple position: subject, predicate or object
* *literals* may appear only in the object position of a triple
* *blank nodes* may appear in the subject or object triple positions only, not in the predicate position

.   | subject | predicate | object
--- | --- | --- | ---
resource   | Y | Y | Y
literal    | X | X | Y
blank node | Y | X | Y

If we recall that *blank nodes* are effectively resources without a URI, the triple position rules are summarised succinctly by the first two rows of the table. In RDF, a triple:
1. always has a *resource* as its subject and predicate
2. may have either a *resource* or *literal* as its object

RDF graph
* an RDF graph is a set of RDF triples
* a set of triples can be viewed as a directed graph in which each triple is an edge from its subject to its object, with the predicate as the label on the edge


## The RDF namespace

The global URI for the RDF namespace is:
`https://www.w3.org/1999/02/22-rdf-syntax-ns#`
* the set of identifiers (constructs) is quite small and is used to define types and properties in RDF

## Common RDF constructs

### rdf:type

This construct is a *property* used to declare that something (an individual) is a member/instance of a set/class
* this property provides an elementary typing system
* it's used as the predicate in a triple
* the object of the triple is understood to be a class (type)
* the subject of the triple can be any *resource* but is understood to be an individual (instance / member) of the class specified in the object of the triple
* this property expresses the relationship between an individual and its class
* hence, it is used for making statements about *individuals*
* in the Turtle syntax for writing RDF it is often abbreviated by the keyword `a`
```RDF
:JoeBloggs rdf:type :Person
lit:Shakespeare rdf:type lit:Playwright
```

### rdf:Property

This construct represents the class of RDF properties
* it is a type (class) used to indicate when a resource (identifier) is to be used as a predicate in an RDF triple rather than as a subject or object
* when one uses this construct to define a property, the property being defined is understood to be an *individual* (instance/member) of the class `rdf:Property`; this is a key reason why we said (above) that RDF, by itself, is mostly used to describe *individuals*; RDF let's us define properties, and use them to describe individuals; but we need RDFS, which extends RDF, in order to be able to describe *properties* (and *classes*) more generally
```RDF
rdf:type rdf:type rdf:Property
:knows rdf:type rdf:Property
:hasAge rdf:type rdf:Property
:JoeBloggs :hasAge :29
lit:wrote rdf:type rdf:Property
lit:Shakespeare lit:wrote lit:KingLear
```



# Inference and the Semantic Web

The Semantic Web is an *inference-based* system.

The meaning of Semantic Web constructs are described in terms of their *inference semantics* (ie the inference rules that they represent).

For the Semantic Web, inferencing means that given some stated information, we can determine other, related information that we can also consider as if it had been stated. Inferencing is a powerful mechanism for dealing with information, and it can cover a wide range of elaborate processing.

Inferencing can:
* make our data more integrated and consistent; eg mundane consistency completion of data
* make our data more connected and consistently integrated

An inference-based system makes data more useful. It can respond to queries based not only on the triples that have been asserted but also on the triples that can be inferred based on the rules of inference.

An RDFS inference engine supports the inferences defined in the RDFS standard; an OWL inference engine supports the larger set of OWL inferences.

There are two main benefits of inference:
1. To infer new triples
2. To find inconsistencies

Specifying when classes are disjoint helps an OWL reasoner to find inconsistencies.



# RDFS

RDFS (RDF Schema)

## Overview

The other Semantic Web standards, like RDFS, build on top of (extend) RDF to provide more modelling capabilities for the *WWW of data*. RDFS is the simplest extension of RDF. The most important capability it provides is that of *inference*: the ability to draw conclusions based on data that has already been seen --- a way to create new data from existing data. Other extensions, like OWL, provide even richer, more advanced *inference* capability.

RDFS gives meaning to RDF data
* is a schema language for RDF
* it lets us describe (provide information about, give meaning to) RDF data
* is itself expressed in RDF
  - ie all schema info in RDFS is defined with RDF triples
* RDFS expresses *meaning* (semantics) through *inference rules* ( inference patterns)
  - we speak of certain RDFS constructs as having *inference semantics*
* these *inference rules* are patterns of triples that permit specific new triples to be inferred and added to the graph
* so, the *meaning* that RDFS gives to *asserted* RDF triples is expressed (manifested / realised) in the new triples that the RDFS *inference rules* enable to be *inferred* and added to the graph
* that is, the meaning of a construct is given by the inferences that can be drawn from it

RDFS is primarily about classes and properties
* it provides guidelines about how to use an RDF graph structure in a disciplined and standardised way
* that is, it is about *classes* and *properties*
* *classes* are sets of individuals
* *properties* are used to:
  - describe individuals  
  - relate individuals to one another
  - relate classes to one another
  - relate properties to one another

RDFS properties (relationships / predicates)
* are first-class, independent resources (entities)
* unlike in OOA/OOD/OOP, RDFS properties are independent of classes
  - a property is never defined 'for a class'
  - classes do not 'have properties'
  - a property can be associated (used) with many classes
* they can be used anywhere, in relation to individuals belonging to any class; (this is the AAA assumption at work)
* we can have triples that describe properties
* in RDF triples, properties can play all 3 roles: subject, predicate and object
  - A) predicate: the predicate property describes the subject or relates the subject to the object
  - B) subject: the triple describes the subject property
  - C) object: the triple relates the subject property to the object property

Examples
* A) a property as predicate:
`:England :capital :London`
  - the predicate (property) is relating one individual to another
  - here, only the predicate is a property
`:Dog rdfs:subClassOf :Animal`
  - the predicate (property) is relating one class to another
  - here, only the predicate is a property
* B) a property as subject:
`:capital rdfs:domain :PopulatedPlace`
  - the triple is describing property `:capital`
  - here, the subject and predicate are properties
* C) a property as object:
`:contractsTo rdfs:subPropertyOf :worksFor`
  - the triple relates the subject property to the object property
  - here, the subject, predicate and object are properties


## RDFS constructs that have inference semantics

`rdfs:Class` - declares something to be a class (type/set)
`rdfs:subClassOf` - relates classes (types/sets) to one another

`rdfs:subPropertyOf` - relates properties to one another

`rdfs:domain` - declares the domain class of a property
`rdfs:range` - declares the range class of a property

## RDFS annotations (constructs without inference semantics)

`rdfs:label` - a common name for a resource (string literal); something that can be used as a printable or human-readable name of a resource

`rdfs:comment` - a comment (info) about a resource (string literal)

`rdfs:isDefinedBy` - primary source of info for a resource (URI)
`rdfs:seeAlso` - supplementary info for a resource (URI)

Annotations:
* RDF triples utilising these RDFS annotation properties are often referred to as **annotations**
* in the Protege ontology editor, all triples using these RDFS properties appear in the *Annotations* window of the GUI, and in exported `.owl/.ttl` files these triples appear in the *Annotations* section of the exported file

## Details of RDFS constructs having inference semantics

### rdfs:Class

`:B rdf:type rdfs:Class`
`:Animal rdf:type rdfs:Class`

### rdfs:subClassOf

`:A rdfs:subClassOf :B`
`:Dog rdfs:subClassOf :Animal`

This property is used to declare how *types* propagate
* it declares that class `:A` is a subclass of class `:B`
* together with `rdfs:Class`, it lets you model class hierarchies
* a class hierarchy lets you propagate types
* it also supports modelling class (set) intersection and class (set) union

Inference semantics: informal
* every member of class (set) `:A` is also a member of class (set) `:B`
* anything of type `:A` is also of type `:B`

Inference semantics: formal
* the *inference semantics* is known as the *type propagation rule*
* expressed in SPARQL, this rule is:
```SPARQL
CONSTRUCT { ?x rdf:type ?B .}
WHERE { ?A rdfs:subClassOf ?B .
        ?x rdf:type ?A }
```
* the same *type propagation rule* automatically applies to every parent class of `:A`, whether declared explicitly or implicitly; eg `:x` will be inferred to be a member of every parent class of `:B`

Multiple parent classes
* a class can be the subclass of many distinct classes
* class `:A` can participate in any number of independent subsumption paths (ie be the subclass of many parent classes)
* there are no 'multiple inheritance'-like problematic issues, as in OOP
* if `:A rdfs:subClassOf :B` and `:A rdfs:subClassOf :C`, the inference semantics of `rdfs:subClassOf` do not change, they are applied twice; so if individual `:x` is a member of class `:A`, then `:x` is also a member of `:B` and `:C`

intersection and union
* `rdfs:subClassOf` can also model the intersection and union of classes (sets)


### rdfs:subPropertyOf

`rdfs:subPropertyOf a rdf:Property`
`:q rdfs:subPropertyOf :r`

This property is used to declare how *predicates* propagate

This property lets you model/build a property hierarchy of related properties
* as with class hierarchies, specific (specialised) properties are at the bottom of the tree and more general properties are higher up in the tree; whenever any property in the tree holds between two entities, so does every property above it
* properties are predicates, relations, relationships
* so a property hierarchy lets you propagate relationships
* relations described for a subproperty also hold for the superproperty
* this property also supports modelling property intersection and property union

* declares that property `:q` is a subproperty of property `:r`
* it makes two properties behave in the same way
* informal meaning: for any triple in which predicate `:q` features, we can replace predicate `:q` with predicate `:r`
* formal meaning: the inference semantics expressed in SPARQL, also referred to as the *predicate propagation rule*, is:
```SPARQL
CONSTRUCT { ?x ?r ?y .}
WHERE { ?q rdfs:subPropertyOf ?r .    
        ?x ?q ?y .}
```

examples:
* given `:brother rdfs:subPropertyOf :sibling` and `:bob :brother :bill`, then we can infer `:bob :sibling :bill`
* `:hasMother rdfs:subPropertyOf :hasParent`

intersection and union
* rdfs:subPropertyOf can model the intersection and union of properties

property transfer
* rdfs:subPropertyOf also supports 'property transfer' --- stating that all uses of P should be considered as uses of Q


### rdfs:domain and rdfs:range

`:p rdfs:domain :D`
`:p rdfs:range :R`

These properties let you link elements of the property hierarchy with elements of the class hierarchy
* they let you specify the domain and/or range classes of a property
* they let you describe how a property is to be used
* they let you specify which classes are used with a property
* they let one describe how a property is used relative to the defined classes

* declares that the domain of property `:p` is class `:D` and the range of property `:p` is class `:R`
* the subject of a triple is classified into the domain of the predicate `:p` and the object of a triple is classified into the range of predicate `:p`
* informal meaning: the property (relation) `:p` relates individuals from class `:D` to individuals from class `:R`; the domain and range classes need not be disjoint or even distinct; whenever we use property `:p`, we can infer that the subject of the triple is a member of class `:D` and the object of the triple is a member of class `:R`

Formal meaning for `rdfs:domain`: the inference semantics expressed in SPARQL
```SPARQL
CONSTRUCT { ?x rdf:type :D .}
WHERE { ?P rdfs:domain ?D .
        ?x ?P ?y }
```

Formal meaning for `rdfs:range`: the inference semantics expressed in SPARQL
```SPARQL
CONSTRUCT { ?y rdf:type :R .}
WHERE { ?P rdfs:range ?R .
        ?x ?P ?y }
```

further points
* if property `:p` is used in a manner inconsistent with the domain/range declarations, no 'error' will be flagged up (raised); instead, RDFS simply infers the 'type' info declared by rdfs:domain and rdfs:range
* domain/range let us draw conclusions about the 'type' of any element based on its use with property `:p`

**combining domain/range with rdfs:subClassOf**

The inference patterns for rdfs:domain, rdfs:range and rdfs:subClassOf can also interact with one another in interesting ways.

For example, whenever we specify `:p rdfs:domain :D`, we can infer that property `:p` also has any superclass of class `:D` as domain too.  The same conclusion holds for rdfs:range.  Thus, the simple definitions of rdfs:domain and rdfs:range are actually quite aggressive.


# OWL - Part 1 - Basics (RDFS-Plus)

This section presents a subset of OWL constructs that the *Working Ontologist* book refers to as *RDFS-Plus*.  These new constructs (features) interact with the features of RDFS to provide a richer modelling environment.

## owl:Class

This is the class of OWL classes
* the OWL specification stipulates that `owl:Class rdfs:subClassOf rdfs:Class`
* this construct is used to declare a resource to be an OWL class
* most ontology software editing tools insist that classes used in OWL models be declared as members of `owl:Class` rather than `rdfs:Class`

Usage:
```RDF
:A rdf:type owl:Class
```

**owl:Class vs rdfs:Class**

`owl:Class` is the class of OWL classes.  It is defined as being a special case (subclass) of `rdfs:Class`, as follows:
```
owl:Class a rdfs:Class .
owl:Class rdfs:subClassOf rdfs:Class .
```
Most OWL ontology editing tools insist that classes used in OWL models be declared as members of `owl:Class`.



## owl:Thing
This is the class of OWL individuals.

## owl:Nothing
This is the empty class. It corresponds to the empty set in set theory.

**Unsatisfiable classes**
Any class that is found to be *unsatisfiable* (ie for which there can be no individuals who are members) is inferred to be a subclass of `owl:Nothing`. This is OWL's way of telling us that the class can have no members.

Once a model contains an unsatisfiable class, it is easy for other class definitions to be unsatisfiable as well.  For example, a subclass of an unsatisfiable class is itself unsatisfiable. A property restriction (on any property) with `owl:someValuesFrom` an unsatisfiable class is itself unsatisfiable. If a property has an unsatisfiable `owl:domain` class or `owl:range` class, then the property becomes basically unusable. The `owl:intersectionOf` two disjoint classes is unsatisfiable. The intersection of any class with an unsatisfiable classes is unsatisfiable.


## owl:inverseOf

This construct relates one property to another property:
* it declares that two properties (relationships) are inverses of one another
* the inverse of a property is another property that reverses its direction
* eg `:hasParent owl:inverseOf :hasChild`
* eg `:ride owl:inverseOf :isRiddenBy`
```RDF
:person :ride :horse
:horse :isRiddenBy :person
:ride owl:inverseOf :isRiddenBy
```

The formal meaning (inference semantics) expressed in SPARQL:
```SPARQL
CONSTRUCT { ?y ?q ?x .}
WHERE { ?p owl:inverseOf ?q .
        ?x ?p ?y }
```
and vice versa. That is, if `:y :q :x`, then infer `:x :p :y`. That is, the inverse relationship is bi-directional.

`owl:inverseOf` has considerable utility in modelling because of how it can interact with other modelling constructs. For example, it is often combined with `rdfs:subPropertyOf`.

Example:
```RDF
:isSonOf rdfs:subPropertyOf :hasParent
:isSonOf owl:inverseOf :hasSon
```
means that from each triple that uses `:hasSon` we can infer a new triple using `:hasParent`. Thus, from `:fred :hasSon :frank` we can infer `:frank :hasParent :fred`.

A *common modelling pattern* is to model a property hierarchy along with a corresponding hierarchy of inverse properties. For example:
```
# property hierarchy
:hasDaughter rdfs:subPropertyOf :hasParent
:hasSon rdfs:subPropertyOf :hasParent
...
# hierarchy of inverse properties
:isDaughterOf rdfs:subPropertyOf :isParentOf
:isSonOf rdfs:subPropertyOf :isParentOf
...
# inverseOf declarations
:hasDaughter owl:inverseOf :isDaughterOf
:hasSon owl:inverseOf :isSonOf
...
```

## owl:SymmetricProperty

This construct declares that a property is its own inverse and, hence, that the property works in both directions (ie is bi-directional)
* ie if `:p` is symmetric, then from `:x :p :y` we can infer `:y :p :x`
* it's an alternate to declaring that `:p owl:inverseOf :p`
* a property being its own inverse is a common enough case that the OWL language provides a special name for this: `owl:SymmetricProperty`
* one asserts that `:p rdf:type owl:SymmetricProperty`

Formal meaning: the inference semantics expressed in SPARQL are
```SPARQL
CONSTRUCT { ?p owl:inverseOf ?p .}
WHERE { ?p rdf:type owl:SymmetricProperty . }
```
and
```SPARQL
CONSTRUCT { ?y ?p ?x .}
WHERE { ?p rdf:type owl:SymmetricProperty .
        ?x ?p ?y }
```


## owl:AsymmetricProperty

This property declares that a property does NOT work in both directions (ie is uni-directional only). If we declare that
```RDF
:parentOf rdf:type owl:AsymmetricProperty
:x :parentOf :y
```
then, if the ontology were extended with the assertion
```RDF
:y :parentOf :x
```
the asymmetric property would be contradicted and the ontology would become inconsistent.


## owl:TransitiveProperty
* is an aspect (characteristic) of a single property
* it's a class of properties
* one asserts that `:p rdf:type owl:TransitiveProperty`
* informal meaning: if R(a,b) and R(b,c), then R(a,c)
* formal meaning: the inference semantics expressed in SPARQL
```SPARQL
CONSTRUCT { ?x ?p ?z .}
WHERE { ?x ?p ?y .
        ?y ?p ?z .
        ?p rdf:type owl:TransitiveProperty .}
```

Examples of transitive properties:
* ancestors/descendants:
* geographic containment: if Osaka is in Japan, and Japan is in Asia, then Osaka is in Asia

Examples of non-transitive properties:
* parents: my parents' parents are not my parents

The *part of* relationship:
* Sometimes it can be somewhat controversial whether a property is transitive or not. And sometimes we may wish to have a property that we can use in either a transitive or non-transitive manner, depending on context.
* For example, the *part of* relationship is sometimes transitive and sometimes not.  Sometimes we might want to use it in a transitive context, sometimes in a non-transitive context.
* transitive case: a toe is part of a foot, and a foot is part of a leg; so a toe is part of a leg
* non-transitive case: Mick Jagger's thumb is part of Mick, and Mick is part of the Rolling Stones; but here we don't want to infer that Mick's thumb is part of the Rolling Stones

**pattern: transitive superproperties**

A powerful modelling pattern involves combining `rdfs:subPropertyOf` and `owl:TransitiveProperty`. This pattern is used in Challenges 18 and 19 of the `Working Ontologist` book (see pgs 243-246).

The pattern:
* We have two properties, one a subproperty of the other. We make the superproperty (only) a transitive property. Triples with property `:p` as predicate lead to the inferencee of triples with property `:q` as predicate. And the newly inferred triples involving `:q` can then trigger, due to the transitivity of `:q`, the inference of further triples (whenever the transitivity inference pattern is matched).
```RDF
:p rdfs:subPropertyOf :q
:q rdf:type owl:TransitiveProperty
```

Consider *part of* relationships. Suppose we wish `dm:partOf` to denote a non-transitive *part of* predicate (property) and `gm:partOf` to denote a transitive version.  Applying the pattern gives  
```RDF
dm:partOf rdfs:subPropertyOf gm:partOf
gm:partOf rdf:type owl:TransitiveProperty
```
Suppose we now use the non-transitive version of the property to state some things about Mick Jagger:
```
:mickThumb dm:partOf :mick
:mick dm:partOf :rollingStones
```
The `dm:partOf` property was NOT declared to be transitive, so the triple `:mickThumb dm:partOf :rollingStones` is NOT inferred. But, triples for the superproperty ARE inferred. The inference semantics of `rdfs:subPropertyOf` mean that triples
```
:mickThumb gm:partOf :mick
:mick gm:partOf :rollingStones
```
are inferred. And, then, because `gm:partOf` is transitive, the inference semantics of `owl:TransitiveProperty` trigger the inference of the additional triple
```
:mickThumb gm:partOf :rollingStones
```


## Equivalence

There are different ways in which two entities can be the same. OWL provides constructs that represent (support) a variety of notions of equivalence.

### owl:equivalentClass
When two classes are known to always have exactly the same members, we say the classes are *equivalent*: `:A owl:equivalentClass :B`.

When two classes are equivalent, it only means that the two classes have exactly the same members. Other properties of the classes are not shared. For example, each class can have its own `rdfs:label`.

Informally, every individual in class A leads to the inference of that individual being a member of class B as well. And vice versa.  That is, the two classes have *exactly* the same members at all times.

The formal meaning: the inference semantics can be most clearly expressed in SPARQL by using two complementary inference patterns/rules
```SPARQL
CONSTRUCT { ?r rdf:type ?B .}
WHERE { ?A owl:equivalentClass ?B .
        ?r rdf:type ?A . }
```
and
```SPARQL
CONSTRUCT { ?r rdf:type ?A .}
WHERE { ?A owl:equivalentClass ?B .
        ?r rdf:type ?B . }
```
Notice that the two rules are effectively simply doing *type propagation*, per the inference semantics of `rdfs:subClassOf`. So `owl:equivalentClass` is really just a bi-directional `rdfs:subClassOf`. So the semantics of `:A owl:equivalentClass :B` can be equivalently described in terms of: `:A rdfs:subClassOf :B` and `:B rdfs:subClassOf :A`.

Another way to view `owl:equivalentClass` is as being inherently symmetric, meaning that `owl:equivalentClass` is its own inverse. Thus, if A equivalent B, then B equivalent A. So whenever `:x :p :y` we can infer that `:y :p :x`.  This observation also captures the bi-directional inference semantics of `owl:equivalentClass`.

If we go further and assert that `owl:equivalentClass rdfs:subPropertyOf rdfs:subClassOf` then we don't need any explicit  inference rules at all --- they can be inferred. To see this, consider an example. Suppose we assert that
```
:Analyst owl:equivalentClass :Researcher
owl:equivalentClass rdfs:subPropertyOf rdfs:subClassOf
```
Then we can infer
```
:Analyst rdfs:subClassOf :Researcher
```
But since `owl:equivalentClass` is symmetric, we can also infer that
```
:Researcher owl:equivalentClass :Analyst
```
which, in turn, lets us infer that
```
:Researcher rdfs:subClassOf :Analyst
```
Given that the pair of triples
```
:Analyst rdfs:subClassOf :Researcher
:Researcher rdfs:subClassOf :Analyst
```
have been inferred, the *type propagation* inference semantics of the `rdfs:subClassOf` takes care of ensuring that the two classes always have exactly the same members, because any triple `:x rdf:type :Analyst` leads to the inference `:x rdf:type :Researcher`, and any triple `:y rdf:type :Researcher` leads to the inference `:y rdf:type :Analyst`.

### owl:equivalentProperty

In the same way that we can use a double (bi-directional) `rdfs:subClassOf` to enforce that two classes are equivalent (have the same set of members), we can use a double (bi-directional) `rdfs:subPropertyOf` to indicate that two properties are equivalent. But OWL provides a more intuitive way to express property equivalence:
```
:p owl:equivalentProperty :q
```

Informally: When two properties are equivalent, any triple that uses one as a predicate leads to the inference of the same triple but with the other property as predicate. This happens in both directions, automatically.

Formally: the inference semantics expressed in SPARQL can be most simply expressed using two rules
```SPARQL
CONSTRUCT { ?a :q ?b .}
WHERE { ?p owl:equivalentProperty ?q .
        ?a :p ?b .}
```
and
```SPARQL
CONSTRUCT { ?a :p ?b .}
WHERE { ?p owl:equivalentProperty ?q
        ?a :q ?b .}
```

We can achieve the same inference outcomes, however, if we make `owl:equivalentProperty` symmetric (which it is), and make it a subproperty of property `owl:subPropertyOf` (which it is):
```RDF
owl:equivalentProperty rdf:type owl:SymmetricProperty
owl:equivalentProperty rdfs:subPropertyOf rdfs:subPropertyOf
```
If we now assert that
```RDF
:borrows owl:equivalentProperty :checkedOut .
```
then, we can use the inference semantics for predicate `rdfs:subPropertyOf` to infer that
```RDF
:borrows rdfs:subPropertyOf :checkedOut
```
and we can use the inference semantics for `owl:SymmetricProperty` to infer that
```RDF
:checkedOut owl:equivalentProperty :borrows .
:checkedOut rdfs:subPropertyOf :borrows .
```
Once we have inferred that `:borrows` and `:checkedOut` are subproperties of one another, we can make all the appropriate inferences.

Thus we see that the rule governing `owl:equivalentProperty` is the same rule as the one that governs `rdfs:subPropertyOf` (except that it works both ways)! By making `owl:equivalentProperty` a subproperty of `rdfs:subPropertyOf`, we explicitly assert that they are governed by the same rule. By making `owl:equivalentProperty` symmetric, we assert the fact that this rule works in both directions.

CAUTION: if you have more than two properties that are mutually equivalent, you have to specify the equivalence of each pair of properties separately; there is no OWL construct to say that a collection of properties are all mutually (pairwise) equivalent.

### owl:sameAs

Property `owl:equivalentClass` declares class equivalence; property `owl:equivalentProperty` declares property equivalence. But when we describe things in the world, we need to do more than describe classes and properties. We also need to describe things themselves --- the members of the classes. We call these *individuals* (instances of classes).

Property `owl:sameAs` is used to declare that two individuals (with different URIs) are equivalent. That is, it declares that two resources with different URIs actually refer to the same thing (the same individual).

Informally, if we say `:a owl:sameAs :b`, then in any triple where we see `:a`, we can infer the same triple with `:a` replaced by `:b`.

Formally, the inference semantics for `owl:sameAs` expressed in SPARQL are defined by 3 rules. The rules cover the cases where the resource in question appears as the triple subject, predicate and object, respectively.
```SPARQL
CONSTRUCT { ?x ?p ?o .}
WHERE { ?x owl:sameAs ?y .  
        ?y ?p ?o}
```
```SPARQL
CONSTRUCT { ?s ?x ?o .}
WHERE { ?x owl:sameAs ?y .  
        ?s ?y ?o}
```
```SPARQL
CONSTRUCT { ?s ?p ?x .}
WHERE { ?x owl:sameAs ?y .  
        ?s ?p ?y}
```

To save us needing 3 further rules with the `owl:sameAs` triples reversed, we also assert that `owl:sameAs` is symmetric, using
```RDF
owl:sameAs rdf:type owl:SymmetricProperty
```
Thus, if `:a owl:sameAs :b`, we can infer that `:b owl:sameAs :a`.


## Functional properties

### owl:FunctionalProperty

A functional property is one that can take one value only for any particular individual (subject), just like a mathematical function can take only one value for a given input. If the property is used for any one individual, then there is exactly one value.

Functional properties are important because they allow *sameness* to be inferred: they provide means of concluding that two resources/individuals (different URIs) refer to the same thing. If a given individual (subject) is observed to have two different values (resource  URIs) for a functional property, then these two values (resources) are inferred to be the same (ie to refer to the same thing and, hence, be the same).

Examples of functional properties:
* `:hasMother`; someone has exactly one mother, but multiple people can share the same mother
* `:hasBirthPlace`; someone has exactly one place of birth
* `:birthDate`; someone has exactly one date of birth

Informally, if `:p rdf:type owl:FunctionalProperty`, then whenever property `:p` is used with two different object resources (URIs), `:a` and `:b`, say, an OWL reasoner infers that `:a` and `:b` must be equivalent.  No 'error' is raised; the reasoner simply infers that the two different URIs refer to the same individual.

Formally, the inference semantics expressed in SPARQL are as follows:
```SPARQL
CONSTRUCT { ?a owl:sameAs ?b .}
WHERE { ?p rdf:type owl:FunctionalProperty .
        ?x ?p ?a .
        ?x ?p ?b . }
```

We see that an `owl:FunctionalProperty` allows two triple *objects* to be inferred as being the same.

Most properties are not functional, but many are. Every time you create a property you should decide whether it is functional or not.


### owl:InverseFunctionalProperty

One can think of an `owl:InverseFunctionalProperty` simply as the inverse of an `owl:FunctionalProperty`. A single value (`:x`) of the property cannot be shared by two individuals. For a property belonging to this class, a given value cannot be shared by two different individuals (subjects). Because of the NUNA (non-unique naming assumption), we cannot tell that two entities are distinct just because they have different URIs. Thus, if two entities are found to share a value for an inverse-functional property, OWL infers that the two entities must be the same (ie that the two URIs must refer to the same individual/thing).

Formally, the inference semantics, expressed in SPARQL, are:
```SPARQL
CONSTRUCT { ?a owl:sameAs ?b .}
WHERE { ?p rdf:type owl:InverseFunctionalProperty .
        ?a ?p ?x .
        ?b ?p ?x . }
```

We see that an `owl:InverseFunctionalProperty` allows two triple *subjects* to be inferred as being the same. If two entities (eg `:a` and `:b`) are found to share a value (eg `:x`) for an inverse functional property, no 'error' is raised. Instead, OWL infers that the two entities (the two triple *subjects*) must be the same.

Examples of inverse functional properties are fairly commonplace:
* any identifying number (social security number, NHS number, student number, employee number, driver's license number, serial number, etc.)
* eg `:hasDiary` --- a person may have many diaries, but it is the nature of a diary that it is authored by one person only

For example:
```RDF
:hasSSN rdf:type owl:InverseFunctionalProperty
:Joe :hasSSN '111-22-333'^^xsd:string
:Hal :hasSSN '111-22-333'^^xsd:string
```
leads to the inference that
```
:Joe owl:sameAs :Hal
```

### 1-to-1 properties: Functional and InverseFunctional

A property can be both an `owl:FunctionalProperty` and an `owl:InverseFunctionalProperty`. This is possible and often very useful. When a property is a member of both of these classes, it is, effectively, a *1-to-1* property. That is, for any one individual there is exactly one value for the property, and for any one value for the property there is exactly one individual.

The analog in mathematics is a function that has an inverse function (ie a *one-to-one* function), where $f(x) = y$ and $f^{-1}(y) = x$. Examples of functions that have inverse functions are monotone functions.

In the case of identification numbers, it is usually desirable that the property be *1-to-1*.  Consider student identification numbers at a university. No two students should share an ID number, and neither should one student be allowed to have more than one ID number.  We can model this in OWL as follows:
* define a property that associates a number with each student and specify its domain and range
* enforce the uniqueness properties using `owl:FunctionalProperty` and `owl:InverseFunctionalProperty`

For example:
```RDF
:hasIdentityNo rdf:type rdf:Property .
:hasIdentityNo rdfs:domain :Student .
:hasIdentityNo rdfs:range xsd:Integer .
:hasIdentityNo rdf:type owl:FunctionalProperty .
:hasIdentityNo rdf:type owl:InverseFunctionalProperty .
```

Now, any two students who share an ID number must be the same (since it is InverseFunctional); furthermore, each student can have at most one ID number (since it is Functional).

Another example:
* `:taxID` is both inverse functional and functional; we want there to be exactly one taxID for each person, and exactly one person per taxID.


## owl:ReflexiveProperty

A reflexive property is one that connects each individual to itself. For example, consider the property `:knows`. A person may know several other people but they also always know themselves. So if we declare
```RDF
:knows rdf:type owl:ReflexiveProperty
```
then, for every individual (eg :Peter), we can infer
```RDF
:Peter :knows :Peter
```
More generally:
```SPARQL
CONSTRUCT { ?a :p ?a .}
WHERE { ?p rdf:type owl:ReflexiveProperty .
        ?p rdf:domain :A .
        ?a rdf:type :A . }
```

## owl:IrreflexiveProperty

An irreflexive property is one where no individual can be connected to itself. For example, consider the property `:marriedTo`. Nobody can be married to themselves. So, if we declare
```RDF
:marriedTo rdf:type owl:IrreflexiveProperty
```
then, if the ontology (KG) were extended with the assertion
```RDF
:Peter :marriedTo :Peter
```
the irreflexivity property would be contradicted and the ontology (KG) would become inconsistent.


## owl:ObjectProperty and owl:DatatypeProperty

These two OWL constructs (classes) do not provide any semantics to a model. But they do provide useful discipline and provide information that many editing tools (like Protege) can take advantage of when displaying models.

OWL provides a way to categorise properties according to the type of the value they can take on.  The object of a triple can be a resource (a URI for a class, property or individual) or a literal. If property `:p` takes a resource as the triple's object, the property is designated as being an `owl:ObjectProperty`; if it takes a literal value, the property is designated as being an `owl:DatatypeProperty`.  In other words, *object properties* relate two objects (resources) to one another; *datatype properties* relate an individual (the subject of the triple) to a literal value. **So object properties relate resources, and datatype properties describe individuals.**

`owl:DatatypeProperty`
* the class of datatype properties
* they describe resources (typically individuals) by specifying particular attributes of that resource/individual
* they relate individuals to literal data (eg strings, numbers, dates, times, etc.)
* more specifically, in a triple (x p y), where *p* is a datatype property, *y* must be a literal
* nb: in Protege, one can specify 'data property assertions' only for an 'individual'; (see the 'Individuals' window and its 'Property assertions' panel)

`owl:ObjectProperty`
* the class of object properties
* object properties relate resources to resources (typically individuals to other individuals)
* more specifically, in a triple (x p y), where *p* is an object property, *y* must be an individual
* nb: in Protege, one can specify 'object property assertions' only for an 'individual'; and the 'object' of the object property assertion must be another 'individual'; (see the 'Individuals' window and its 'Property assertions' panel)

For example:
```RDF
bio:married rdf:type owl:ObjectProperty .
:AnneHathaway bio:married lit:Shakespeare .

mfg:Product_SKU rdf:type owl:DatatypeProperty .
mfg:Product1 mfg:Product1_SKU "FB3524" .
```


# OWL - Part 2 - Defining Classes - Property Restrictions

Here we discuss the topic of Restrictions (or Restriction classes). In particular, we focus on property restrictions --- that is, restrictions placed on properties that describe the class of individuals (the set of things) that satisfy the restriction.

## Property Restrictions - overview

*Restrictions* let us describe classes (sets of things) in terms of other things we have already modelled (properties and classes). This opens up whole new vistas in modelling capabilities.

A *Restriction* lets us define a class (set) of individuals. It lets us describe a class (set) of individuals in terms of a specified condition shared by those individuals: a property possessed by those individuals and the value(s) taken by that property.

A *Restriction* is a special case of a class. They are sometimes referred to as *Restriction classes*. The intuition behind the name *Restriction* is that **membership in the new class is *restricted* to those individuals that satisfy a specified condition (ie have a particular characteristic)**.  In other words, the *Restriction class* consists of the set of individuals that satisfy the condition (possess the characteristic).

A *Restriction* in OWL is a class defined by describing the individuals it contains --- ie by describing the characteristics that individual members of the class possess. If you can describe a set of individuals in terms of known classes and properties, then you can use that description to define a new class, a *Restriction class*. In other words, a *Restriction* is a new class whose description is based on descriptions of its propective members.  That is, a *Restriction* is a class that is defined by a description of its members in terms of existing classes and properties. A *Restriction* is defined by providing some description that limits (or restricts) the kinds of things that can be said about a member of the class.  

A *Restriction class* in OWL utilises the keyword (construct) `owl:onProperty`. This specifies what property is to be used in the definition of the restriction class. It identifies the *designated property* upon which a restriction is being placed.

The definition of a *Restriction class* also defines the precise *nature* of the restriction being placed on the values of the designated property.

Such *Restriction classes* are also often referred to as **property restrictions**.

OWL provides a number of kinds of property restrictions, each of which places different restrictions on the values taken on by the property being restricted (the designated property). Three of these kinds of restrictions are: `owl:allValuesFrom`, `owl:someValuesFrom` and `owl:hasValue`. Each describes how the new class being described is constrained by the possible asserted values that the designated property can take on.

A restriction is a special kind of class, so it has individual members just like any class. Members (individuals) of a *Restriction class* must satisfy the conditions specified by the restriction:
1. they must be described by (ie be the subject of) a triple whose predicate is the restricted property specified by `owl:onProperty`, and
2. the value(s) of that property must match the kind of restriction (`owl:allValuesFrom`, `owl:someValuesFrom` or `owl:hasValue`) being placed on the property.

Individuals that satisfy the conditions specified by (the description of) a *Restriction class* are inferred to be members of that class.

Two main things make a *property restriction* class different from other OWL classes:
1. it's a class defined by a property that its member have, and
2. it *need not have a name* because it can be used purely as an expression that's part of some larger expression (eg intersection or union of classes).

The property designated in the *Restriction class* definition can be either an *object property* or *data property*. A data property simply connects individuals to a literal rather than another individual.

Description Logics
* NOTE: Logics that provide flexible means (such as this) for defining classes in terms of complex descriptions of the members of the class are called *Description Logics*. The word *Description* in the name refers to the ability to define classes via arbitrarily complex descriptions, rather than simply by naming concepts (like Table, Horse, Person, etc.).


## owl:Restriction

The `owl:Restriction` construct is the class of property restrictions. It is used to declare and define an OWL *Restriction*.

## owl:onProperty

The `owl:onProperty` property designates the property to which a property restriction refers.  

## owl:someValuesFrom

This construct (property) is used to create a restriction of the form "All individuals for which at least one value of the property P comes from class C". The value of property P is restricted to individuals of class C.

In logic, this kind of restriction corresponds to the symbol $\exists$, referred to as the *existential quantification* operator, and interpreted as saying "There exists ..." (which is equivalent to "one or more" and "at least one").  Note that the Protege ontology editor uses the keyword *some* to represent construct `owl:someValuesFrom` in its object property and data property editor views.

An example *Restriction class* (property restriction) for this kind of restriction is:
```RDF
[ rdf:type owl:Restriction ;
  owl:onProperty :P ;
  owl:someValuesFrom :C .]
```

Notice that a restriction class is defined using a blank node (an anonymous node or bnode). This bnode is the subject of each triple in the definition of the restriction class. This bnode is described by the properties listed in the restriction.

A restriction class defined in this way refers to exactly the class (set) of individuals that satisfy the conditions specified.

A restriction class such as this has no specific name associated with it. It is defined by the properties of the restriction. It is sometimes referred to as an "unnamed class" or an "anonymous class".

## owl:allValuesFrom

This construct (property) is used to create a property restriction class of the form "the individuals for which all values of the property P come from class C". The value of property P is restricted to individuals of class C.

In logic, this kind of restriction corresponds to the symbol $\forall$, referred to as the *universal quantification* operator, and interpreted as saying "For all ...".  An alternate, equivalent way of expressing "for all" is via the word "only".  For example: "the individuals for which property P has values coming from class C only".  Note that the Protege ontology editor uses the keyword "only" to represent construct `owl:allValuesFrom` in its object property and data property editor views.

An example restriction class for this kind of restriction is
```RDF
[ rdf:type owl:Restriction ;
  owl:onProperty :P ;
  owl:allValuesFrom :C .]
```

Again, a restriction class defined in this way refers to exactly the class (set) of individuals that satisfy the conditions specified.

Note that, in logic, the "for all" condition (ie universal quantification) is satisfied when an individual has zero of the specified property values.  For example, if a person has no children, then it is valid to say that 'all' of his or her children are boys (or girls).

So individuals who are NOT the subject in any triples using the property being restricted actually satisfy the 'universal quantification' condition, and hence will be members of the restriction class. More concretely, if an individual `:x` has no triple `:x :P ...`, then this individual satisfies the conditions of the 'for all' restriction class.

In contrast, `owl:someValuesFrom` (existential quantification) guarantees that *some* value (at least one) is given for the specified property. But `owl:allValuesFrom` cannot guarantee that a triple exists with the specified property, because zero values for the specified property also satisfy the universal quantification condition.

## owl:hasValue

This construct (property) is used to create a restriction class of the form "All individuals that have the value A for the property P".  Property P is restricted to a single, specific value: A.  If property P is an object property (`owl:ObjectProperty`), this value is an individual (a URI for a resource); if property P is a data property (`owl:DatatypeProperty`), this value is a literal.

It specifies a single value for a property.  That is, it describes the set of individuals that have a specified property taking a specified single value.

An example restriction class for this kind of restriction is
```RDF
[ rdf:type owl:Restriction ;
  owl:onProperty :P ;
  owl:hasValue A .]
```

This is really just a special case of the `owl:someValuesFrom` restriction, in which the class C happens to be a singleton set A.  But it is a useful modelling form. It effectively turns specific instance descriptions into class descriptions.

Note that `owl:hasValue` not only guarantees that a triple exists with the specified property, it even specifies exactly what that triple is.  That is, if A is a member of the property restriction `owl:onProperty :P` `:hasValue X`, then we can infer the triple `:A :P X`.

## Giving names to Property Restriction classes

The examples of *property restriction* classes given above are unnamed, anonymous classes: the *bnode* (blank node) of the *Restriction class* is not connected to any other named class in any way. This may or may not be sufficient.

Whether or not to give a *property restriction* class an explicit name is a modelling decision. The main reason to give a property restriction class a name is if it needs to be used in more than  situations.

Sometimes an anonymous class can be used 'as is' (eg in an intersection or union expression).  At other times, to be usable, a restriction class needs to be given an explicit name (ie be connected to another named class). This is done via either the `rdfs:subClassOf` or `owl:equivalentClass` properties.  If a *Restriction class* is associated with an explicit name then one can refer to that named class elsewhere in the ontology as often as required.

The choice of how link the anonymous *Restriction class* with another named class (via either the `rdfs:subClassOf` or `owl:equivalentClass` property) depends on the nature of the inferences you wish to be able to make (ie whether they need to be uni-directional or bi-directional):
* `rdfs:subClassOf` supports uni-directional inferencing only:
  - if `:A rdfs:subClassOf :B` and `:x rdf:type :A`, then we can infer `:x rdf:type :B` only
* `owl:equivalentClass` supports bi-directional inferencing because it's a symmetric property (ie it is its own inverse).

Example of naming a *Restriction class*:
```RDF
q:SelectedAnswer rdfs:subClassOf
    [ rdf:type owl:Restriction ;
      owl:onProperty q:enablesCandidate ;
      owl:allValuesFrom q:EnabledQuestion ] .
```


# OWL - Part 3 - Defining Classes - Set Theory

In this section we cover OWL constructs that represent a full set theory language, including set unions, intersections and complements, as well as set disjointness.

As we will see, these set theory constructs are used to create new classes by combining other classes, including *property restriction classes*. The ability to combine classes afforded by these set-theoretic constructs provides a potent system for making very detailed descriptions of knowledge/information.

## Unions and Intersections

OWL provides a facility for defining new classes as *unions* and *intersections* of previously defined classes (whether regular classes or property restriction classes).

### owl:unionOf

The union of two or more classes includes the members of all those classes combined.

A union class can be defined in two ways: 1) by using `owl:unionOf` to define your new union class directly, or 2) by using `owl:unionOf` to first define an anonymous class and then declaring that anonymous class to be equivalent to your new union class using `owl:equivalentClass`. Note that the Protege ontology editor uses method (2). When one defines an 'Equivalent To' entry in the 'Description' frame of a class in the Class hierarchy, Protege uses method (2) when exporting the ontology to a text file.

Usage - direct naming:
```RDF
:UC rdf:type owl:Class .
:UC owl:unionOf (:A :B :C) .
```

Usage - anonymous class:
```RDF
:UC rdf:type owl:Class .
:UC owl:equivalentClass [ rdf:type owl:Class ;
                          owl:unionOf (:A :B :C)
                        ] .
```


### owl:intersectionOf

The intersection of two or more classes includes the members that belong to every one of the classes.

An intersection class can be defined in two ways: 1) by naming it directly, 2) by defining an anonymous class and declaring it to be equivalent to a named class using `owl:equivalentClass`.

Usage - direct naming:
```RDF
:IC rdf:type owl:Class .
:IC owl:intersectionOf (:A :B :C) .
```

Usage - anonymous class:
```RDF
:IC owl:equivalentClass
    [ rdf:type owl:Class ;
      owl:intersectionOf (:A :B :C)
    ] .
```

### using these constructs with restriction classes

Unions and intersections work just as well on property restriction classes as on regular classes. In fact, using them with restriction classes allows one to do more complex modelling than would otherwise be feasible.

For example, suppose we wish to model the class of "all planets orbiting the sun". This natural language phrase actually reflects an intersection --- of 1) all planets, and 2) all things that orbit the sun. Observe that set (2) is actually a property restriction class; it's a `:hasValue` restriction ('the sun') on property 'orbits'.

One way to model this class (set) of "all planets orbiting the sun" in OWL is to use the 'direct naming' approach, as follows:
```RDF
:SolarPlanet rdf:type owl:Class ;
:SolarPlanet owl:intersectionOf (
                      :Planet
                      [ rdf:type owl:Restriction ;
                        owl:onProperty :orbits ;
                        owl:hasValue :TheSun ]
             ) .
```

## Enumerating the members of a class

Here we describe another way to define a class. This way involves explicitly listing (enumerating) the individual members of the class.  

Defining classes in this way represents a small 'closure' of the open world of the Semantic Web. Recall that the Open World Assumption implies that a new fact can be discovered at any time. If we specify the precise members of a class, the class is closed to new members. However, there are reasons for wanting the ability to do this on occasion.  An open world where new facts can be discovered at any time complicates inferences regarding set complements and inferences that depend on counting things (like cardinality restrictions). Inferences having to do with set complements and counting are much clearer if one first asserts that certain parts of the world are closed.

OWL provides the ability to define a class by specifying its individual members so as to enable the modelling of certain closed aspects of the world, for example when inferencing involving set complements are counting (cardinality restrictions) are required.

### owl:oneOf - enumerating the members of a class

When one is in a position to enumerate the members of a class (set), a number of inferences can follow. OWL lets us enumerate the members of a class using the construct `owl:oneOf`.

This is the property that determines the collection of individuals or data values that build an enumeration. Its domain is a class; its range is an RDF list (rdf:List) --- a list of the members of the class.

For example, we can define the class of planets of our solar system as follows:
```RDF
:SolarPlanet rdf:type owl:Class .
:SolarPlanet owl:oneOf (:Mercury :Venus :Earth ... :Neptune) .
```

Informally, the meaning of class `:SolarPlanet` is that it contains the 8 named planets (individuals) and no others.  That is, the class is made up of exactly these 8 items, and no others.

When we say a class is made up of exactly these items, nobody else can say that there is another distinct item that is a member of that class. That is, `owl:oneOf` places a limit on the "Anyone can say Anything about Any topic" (AAA) slogan. Thus, `owl:oneOf` should be used with care and only in situations in which the definition of the class is not likely to change.

Whereas `:hasValue` specifies a single value for a property, `owl:someValuesFrom` combined with `owl:oneOf` specifies a distinct set of values for a property.

The meaning of `owl:oneOf` goes further than simply asserting the members of a class; it also asserts that these are the *only* members of this class. Thus, if we later assert that some new individual is a member of the class, then that individual must be equivalent to (ie `owl:sameAs`) one of the members listed in the `owl:oneOf` list.

## Set complement

The complement of a set is the set of all things not in that set. The same definition applies to classes in OWL. The complement of a class is another class whose members are all the things not in the complemented class.

### owl:complementOf

A complement always applies to a single class. We can define one as follows:
```RDF
:ClassA owl:complementOf :ClassB
```

Set complements can easily be misused, and OWL can be quite unforgiving in such situations. For example, the complement of class (set) `:MajorLeaguePlayer` is actually everything else in the universe other than major league players.

### combining it with owl:intersectionOf

To avoid situations like the one just described, it is common practice to not refer to complementary classes directly. Instead, it is common practice to combine complement with intersection. For example, we can define a *minor league player* as a *player* who is not a *major league player* by asserting:
```RDF
:MinorLeaguePlayer owl:intersectionOf (
    :Player
    [ rdf:type owl:Class ;
      owl:complementOf :MajorLeaguePlayer
    ]
)
```
This definition makes use of a blank node to specify an anonymous class. There is no need to name the class that is the complement of `:MajorLeaguePlayer`, so it is specified anonymously using the *bnode* notation `[ ... ]`.


# OWL - Part 4 - Differentiating Things

Here we discuss how OWL allows one to differentiate individuals.

The act of differentiating things (of declaring them to be distinct) is another way of 'closing' the world a bit, by specifying that the non-unique naming assumption does not apply for two (or more) specific resources.  That is, we state that the designated URIs refer to different individuals.

As mentioned earlier, closing the world a bit in can yield benefits in terms of making it feasible to precisely determine set complements and to precisely count (eg which is what cardinality restrictions must do).   

## Differentiating individuals

### owl:differentFrom - differentiating individuals

Due to the Non-unique Naming Assumption (NNA) in the Semantic Web, we have to state explicitly when two things are, in fact, different from one another. OWL provides the construct (property) `owl:differentFrom` this purpose.  This is the property that determines that two given individuals are different.

To assert that one resource is different from another, we say
```RDF
:Earth owl:differentFrom :Mars .
```

As we will see, `owl:differentFrom` supports a number of inferences when used in conjunction with other constructs like `owl:cardinality` and `owl:oneOf`.

### owl:AllDifferent and owl:distinctMembers - differentiating multiple individuals

Specifying a list of items, all of which are different from one another, soon gets awkward. If we have a class (set) with K members, there are $\frac{K(K-1)}{2}$ distinct pairs of members. So, for a set with 8 members, to declare explicitly that the members in each distinct pair of members are different from one another, we would need 28 `owl:differentFrom` assertions.  

To simplify things, OWL provides the `owl:AllDifferent` class and the `owl:distinctMembers` property, two constructs which are used together to specify explicitly, but succinctly, that the individuals in a list are distinct from one another. By using these two constructs, we specify that the list should be treated as a set of mutually different individuals.

We use property `owl:distinctMembers` to indicate that a list should be treated as a set of mutually different individuals. The domain of `owl:distinctMembers` is class `owl:AllDifferent`.  That is, the subject of `owl:distinctMembers` must be a member (individual) of class `owl:AllDifferent`.

It is customary for the subject of an `owl:distinctMembers` triple to be a *bnode* (blank node). For example, we would define a class (set) with K distinct individuals by saying
```RDF
[ rdf:type owl:AllDifferent ;
  owl:distinctMembers (:A1 :A2 :A3 ... :AK)
] .
```

Formally, with $K=8$, this is equivalent to asserting 28 `owl:differentFrom` triples, one for each pair of individuals in the list.

Recall that we can use `owl:oneOf` to specify a list of individuals and, crucially, indicate that these are the *only* members in existence. Now we have gone further to say that additionally these individuals are distinct.  It is quite common to use `owl:oneOf` and `owl:AllDifferent` together in this way to say that a class is made up of an enumerated list of distinct elements. For example, one can also say things such as:
```RDF
:JamesDeanMovie rdf:type owl:Class .
:JamesDeanMovie owl:oneOf (:Giant :EastOfEden :Rebel) .
:JamesDeanMovie rdf:type owl:AllDifferent
```

## Differentiating classes (disjoint sets)

### owl:disjointWith

Disjoint sets are sets that have no individual in common. In OWL we represent disjoint classes (sets) using the property `owl:disjointWith`.

Example:
```RDF
:Reptile owl:disjointWith :Mammal .
```

For any members of disjoint classes, we can infer that they are `owl:differentFrom` one another. For example, if we assert that
```RDF
:Lassie rdf:type :Mammal .
:Godzilla rdf:type :Reptile .
```
then we can infer that
```RDF
:Lassie owl:differentFrom :Godzilla .
```

This simple idea can have powerful ramifications when combined with other OWL constructs.

### owl:AllDisjointClasses and owl:members

Just as we have `owl:AllDifferent` as a way to specify that several individuals are mutually distinct, we have `owl:AllDisjointClasses` to indicate that the classes in a group of classes are all mutually disjoint. It is used in combination with property `owl:members`, which specifies the collection (list) of classes under consideration.

For example, we can say
```RDF
:DisjointMovies rdf:type owl:AllDisjointClasses .
:DisjointMovies rdfs:label "Disjoint movies" .
:DisjointMovies owl:members (
    :MovieClassA
    :MovieClassB
    :MovieClassC
) .
```

## Differentiating properties (disjoint properties)

### owl:propertyDisjointWith

Description:
* The property that determines that two given properties are disjoint.
* two properties are disjoint if no two individuals are interlinked by both properties
* two individuals cannot be linked by both properties at the same time
* this construct allows us to state that two individuals cannot be related to each other by two different properties that have been declared disjoint
* `:p1 owl:propertyDisjointWith :p2`
* eg: state that parent-child marriages cannot occur:
`:hasParent owl:propertyDisjointWith :hasSpouse .`
  - ie we cannot have `:x :hasParent :y` and `:x :hasSpouse :y`
* applies between two object properties and between two data properties

### owl:AllDisjointProperties

It's a convenient way to declare that the properties in a collection are all mutually, pairwise disjoint with respect to one another.

The construct works the same way as `owl:AllDisjointClasses` does for classes, except in relation to properties. It works with `owl:members` in the same was as does `owl:AllDisjointClasses`.



# OWL - Part 5 - Defining Classes - Cardinality Restrictions

Here we discuss another way of defining new classes (sets of individuals). These methods involve another type of OWL restriction: cardinality restrictions.  Cardinality restrictions refer to the number of distinct values individuals have for a particular property. For example, we can describe "the set of planets that have at least 3 moons".

As we will see, reasoning with cardinalities in OWL is surprisingly subtle. As a consequence, cardinality inferencing in OWL is quite conservative in the conclusions it can draw.

Recall that *property restrictions* define classes based on the presence of certain values for given properties.  OWL also allows one to define restriction classes based on the number of distinct values a given property takes.  These are called *cardinality restrictions*.  Cardinality restrictions allow us to express constraints on the number of individuals that can be related to a member of the cardinality restriction class.

For example, a person has two (biological) parents. A baseball team has exactly nine (distinct) players in its starting lineup.

Cardinality restrictions can be used to define fine-grained sets of particular interest.

Cardinality always refers to the number of *distinct* values a property has; it therefore interacts closely with the Non-unique Naming Assumption and `owl:differentFrom`.

## owl:cardinality - (exactly)

Example: this cardinality restriction defines the class of things that have *exactly* nine distinct values for a given property:
```RDF
:NinePlayerTeam owl:equivalentClass
    [ rdf:type owl:Restriction ;
      owl:onProperty :hasPlayer ;
      owl:cardinality 9 ]
```

If we can prove that an individual has exactly N distinct values for property P, then it is a member of the corresponding  `owl:cardinality` restriction class.

## owl:minCardinality - (at least)

Example: this cardinality restriction defines the class of things that at least some (lower bound) number of values for a given property:
```RDF
[ rdf:type owl:Restriction ;
  owl:onProperty :hasPlayer ;
  owl:minCardinality 10 ]
```

## owl:maxCardinality - (at most)

Example: this cardinality restriction defines the class of things that at most some (upper bound) number of values for a given property:
```RDF
[ rdf:type owl:Restriction ;
  owl:onProperty :hasPlayer ;
  owl:maxCardinality 2 ]
```

## Qualified Cardinality Restrictions

OWL 2.0 supports *qualified cardinality*.  This lets us go further than specifying the number of distinct values for a given property; it lets us specify how many values should be from a particular class. The concept of qualified cardinality thus permits more detailed cardinality information to be described. Having this capability can be essential, especially when modelling teh structure of complex objects.  Structural models often make extensive use of qualified cardinalities.

A simple example of qualified cardinality is a model of a hand: a hand has 5 fingers, one of which is a thumb.  Another example is a loan. A loan is a contract with two parties, but one is a lender and the other a borrower.

One defines *qualified cardinality restriction* classes in the same way that one defines *cardinality restriction* classes except that the definition includes a reference to property `owl:onClass`.

### owl:onClass

The `owl:onClass` property determines the class that a qualified object cardinality restriction refers to.

### owl:qualifiedCardinality - (exact)

### owl:minQualifiedCardinality - (at least)

For example, suppose a domain with movies and actors and awards. Suppose we have the following classes for Academy Award winning actors:
```RDF
:BestActor rdf:type owl:Class .
:BestSupportingActor rdf:type owl:Class .
:OscarWinningActor owl:unionOf (:BestActor :BestSupportingActor) .
```
Now, suppose we wish to define the class of 'star-studded movie', the members of which we wish to be movies with at least 2 Oscar winning actors. We can define this class using *qualified cardinality* as follows:
```RDF
:StarstuddedMovie owl:intersectionOf (
  :Movie
  [ rdf:type owl:Restriction ;
    owl:onClass :OscarWinningActor ;
    owl:onProperty :stars ;
    owl:minQualifiedCardinality 2 . ]
  )
```

### owl:maxQualifiedCardinality - (at most)


## Small cardinalities

Here we consider four special cases of cardinality --- ones that arise when the cardinalities are limited to the small numbers 0 and 1.

**minCardinality 0**
Specifying the restriction `owl:minCardinality 0` describes a set of individuals for which the presence of a value for the specified `owl:onProperty` is *optional*.  In the semantics of OWL, this is superfluous since properties are always optional anyway. But the explicit assertion that something is optional can be used for model readability.  In fact, this restriction is equivalent to `owl:Thing`, the class of all individuals. So it's not much of a restriction.

**maxCardinality 0**
This cardinality restriction indicates the set of individuals for which no value for the specified property is allowed.

**minCardinality 1**
This cardinality restriction indicates the set of individuals for which some value for the specified property is present. This is subtly different from a `owl:someValuesFrom` (existential quantification) property restriction class because the cardinality restriction refers only the values of a property, whereas the property restriction refers to values of a property from a specified class.

**maxCardinality 1**
This cardinality restriction specifies that a value is unique (but need not exist). Its the set of individuals with at most 1 value for a given property.

**final comments**
Considering `minCardinality 1` and `maxCardinality 0`, one can observe that:
* they are disjoint from one another
* they are both subclasses of `minCardinality 0`
* their union makes up all of `minCardinality 0`; so they form a partition of `minCardinality 0`.


# OWL - Part 6 - Negative Property Assertions

Negative property assertions allow one to express/model negative facts: fact negation. A negative property assertion states that an individual *cannot possibly* have a certain value for a certain property.

Semantics:
* fact negation
* stating that something is NOT the case
* stating that one individual is NOT related to another value (individual or literal) by a certain property
* stating that the relation of an individual to either data or another individual via a certain property *does not exist*
  - ie it's not just the case that no such triple currently exists in the KG; no such triple exists (and never will exist in the KG without the KG becoming inconsistent).
* an implication of the Open World Assumption is that if an individual is not linked by a certain property with a certain value (an individual or a literal), there are multiple candidate explanations: 1) the individual really does not have the property with the specified value, or 2) the individual does have the property with the value but this fact is currently unknown because the information is simply missing from the KG, or 3) the individual currently does not have the property with the value, but it could have at any point in the future. A negative property assertion states that the individual *cannot possibly have that property value*.  The underlying reason may be that the property itself is invalid with respect to the individual, or that the property is valid but the value is invalid with respect to the individual.

One can express both *negative object property assertions* and *negative data property assertions*.  The former uses `owl:targetIndividual`, the latter uses `owl:targetValue`.

[In Protege, see the 'Property assertions' panel of the 'Individuals' window for where to define negative object/data property assertions.]

Several OWL constructs are involved in defining negative property assertions.

## owl:NegativePropertyAssertion

This construct is the class of negative property assertions.

Example of a *negative object property assertion*
* individual :s is NOT related to individual :t by object property :op
* or, equivalently: individual :s does NOT (cannot) have property :op with individual :t
```RDF
:x rdf:type owl:NegativePropertyAssertion
:x owl:sourceIndividual :s
:x owl:assertionProperty :op
:x owl:targetIndividual :t
```

Example of a *negative data property assertion*
* individual :s cannot have value :t on data property :dp
```RDF
:x rdf:type owl:NegativePropertyAssertion
:x owl:sourceIndividual :s
:x owl:assertionProperty :dp
:x owl:targetValue :t
```

## owl:sourceIndividual

The property that determines the subject of a negative property assertion.

## owl:assertionProperty

The property that determines the predicate of a negative property assertion.

## owl:targetIndividual

The property that determines the object of a negative object property assertion.

## owl:targetValue

The property that determines the value of a negative data property assertion.



# OWL - Part 7 - Other OWL 2.0 Constructs

## owl:hasKey

The property that determines the collection of properties that jointly build a key:
* a set of properties can be associated with a class through `owl:hasKey`
* two members of the class are considered to be the same (`owl:sameAs`) if they have the same values for all the properties making up the key
* the relational database analogy is the set of fields making up the primary key for a relational table
* if we define the properties `:firstName`, `:lastName` and `:address` for the class `:Person`, then two people would be considered the same whenever all of these properties have matching values

## owl:NamedIndividual

This is the class of named individuals.  New with OWL 2.0.

This property declares that a resource is an individual.

One explicitly declares a resource to be a class with `owl:Class`. One explicitly declares a property to be a property with `rdf:Property`.

Every individual is necessarily an instance of `owl:Thing`. Every individual that you give a name to (ie a URI) is an instance of `owl:NamedIndividual`. The ontologist normally does not have to specify these types; they are normally created automatically by one's ontology editor tool.

For example, the Pizza ontology `pizza.owl` used for the Protege Pizza tutorial contains definitions of 5 individuals. All are declared to be NamedIndividuals.
```RDF
<owl:Thing rdf:about="http://www.co-ode.org/ontologies/pizza/pizza.owl#Italy">
    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#NamedIndividual"/>
    <rdf:type rdf:resource="http://www.co-ode.org/ontologies/pizza/pizza.owl#Country"/>
</owl:Thing>
```

If I go into Protege and define two individuals and associate the first property with a particular class, and then save the ontology (in Turtle format), I get:
```RDF
:myindivid1 rdf:type owl:NamedIndividual ,
                     :myclass1 .

:myindivid2 rdf:type owl:NamedIndividual .
```



## owl:disjointUnionOf

The property that determines that a given class is equivalent to the disjoint union of a collection of other classes.

## owl:onProperties

The property that determines the n-tuple of properties that a property restriction on an n-ary data range refers to

## owl:propertyChainAxiom

The property that determines the n-tuple of properties that build a sub property chain of a given property.




# SHACL

SHACL (Shapes Constraint Language)

## Introduction

Like OWL, SHACL builds on top of (extends) RDFS to provide more modelling capabilities for the *WWW of data*. But whereas RDFS and OWL focus on providing capabilities for *inference*, the focus of SHACL is to provide a modeler with capability for managing *expectation*. *Expectation* involves forming some prediction about data we haven't yet seen. The need to express and manage expectation can come into play when eliciting data from a user or validating data from a new data source. SHACL is the extension of RDFS that allows a modeler to manage expectation.

SHACL is the *expectation modelling language* of the Semantic Web. SHACL provides a mechanism for specifying the expected shape of a data description and constraints on its validity.

One of the basic assumptions of the Semantic Web is the OWA (Open World Assumption) where we only draw conclusions from data that will not be undermined or countermanded by the discovery of new data.  This assumption makes sense when we are interpreting data we find 'in the wild', like data on the Web. But there are some situations in which we are not interpreting data, we instead are acting on some *expectations* about the data in the wild. Such expectations might come from regulatory control or business policies or just plain common sense.

Expectations in the Semantic Web can take 3 broad forms:
1. Data validation.
  - we want to validate that data matches our expectation
2. Soliciting user input.
  - we may be collecting data from users and so wish to communicate our expectations regarding the data they will be providing; this is typically done via features/wizards of a GUI/HTML form through which the users are supplying information  
3. Validating user input.
  - despite having communicated our expectations of data to users, we may still wish to validate what they give us against our expectations

We can think of our expectations for data as a specification of the *shape* that the data should take. When soliciting user input, we can literally use this specification to create a form for users to fill out. The metaphor *shape* is more figurative in the case of data validation.

Our expectations, expressed in a description of data, can specify:
* the *shape* of the data
  - eg a SIN has a particular structure XXX-XX-XXXX where X in [0-9]
* *constraints* on data values
  - eg cardinalities (min/max number of values)
  - eg value ranges/sets; eg 'gender' in ['male','female','other']

Like RDFS and OWL, SHACL is expressed in RDF.



# Common Class Inference Patterns

OWL allows us to draw a wide range of conclusions about classes. For example, in some circumstances, we can infer that one class is a subclass of another or that a class is the domain (or range) of a property. Here we present a few common patterns that are worth calling out.

## Intersection and subclass

The intersection of two (or more) classes is a subclass of each intersected class. So, if
```RDF
:A owl:intersectionOf (:B :C) .
```
then we (or an OWL reasoner) can infer
```RDF
:A rdfs:subClassOf :B .
:A rdfs:subClassOf :C .
```

## Union and subclass

The union of two (or more) classes is a superclass of each class in the union. So, if
```RDF
:A owl:unionOf (:B :C) .
```
then we (or an OWL reasoner) can infer
```RDF
:B rdfs:subClassOf :A .
:C rdfs:subClassOf :A .
```

## Complement and subclass

*Complement* reverses the order of subclass. For example, if
```RDF
:A rdfs:subClassOf :B .
```
and
```RDF
:Ac owl:complementOf :A .
:Bc owl:complementOf :B .
```
then we (or an OWL reasoner) can infer
```RDF
:Bc rdfs:subClassOf :Ac
```

## owl:someValuesFrom, owl:allValuesFrom and subclass

Subclass relationships persist through corresponding property restriction classes. If A is a subclass of B, then a restriction on property P with respect to class A is a subclass of the same restriction on property P with respect to class B. This holds for restriction constructs `owl:someValuesFrom` and `owl:allValuesFrom`.

More formally, consider the property restriction `owl:someValuesFrom`. If we have
```RDF
:A rdfs:subClassOf :B .
```
and we have this property restriction on property P wrt class `:A`
```RDF
:PrA owl:equivalentClass
    [ rdf:type owl:Restriction ;
      owl:onProperty :P ;
      owl:someValuesFrom :A . ]
```
and we have this property restriction on property P wrt class `:B`
```RDF
:PrB owl:equivalentClass
    [ rdf:type owl:Restriction ;
      owl:onProperty :P ;
      owl:someValuesFrom :B . ]
```
Then we can infer
```RDF
:PrA rdfs:subClassOf :PrB .
```

This same propagation principles holds for *any* property and also for `owl:allValuesFrom` restrictions.

## owl:hasValue, owl:someValuesFrom and subclass

The relationship between property restriction `owl:hasValue` and `rdfs:subClassOf` works a bit differently. Suppose an individual `:x` is a member of class `:A`. Then the restriction `owl:hasValue :x` on property P is a subclass of the restriction `owl:someValuesFrom :A` on property P.

More formally, if we have
```RDF
:x rdf:type :A .
```
and
```RDF
:Prx owl:equivalentClass
    [ rdf:type owl:Restriction ;
      owl:onProperty :P ;
      owl:hasValue :x . ] .
```
and
```RDF
:PrA owl:equivalentClass
    [ rdf:type owl:Restriction ;
      owl:onProperty :P ;
      owl:someValuesFrom :A . ] .
```
then we can infer
```RDF
:Prx rdfs:subClassOf :PrA .
```

## Relative cardinalities and subclass

Subclass relations between cardinality restrictions arise from the usual rules of arithmetic on whole numbers. For example, suppose a cardinality restriction class A whose members have at least 9 values on property P (owl:minCardinality 9) and suppose another cardinality restriction class B whose members have exactly 10 values on property P (owl:cardinality 10). Then we can infer that B is a subclass of A, ie that `:B rdfs:subClassOf :A`.

## owl:someValuesFrom vs owl:allValuesFrom

In logic, existential quantification $(\exists)$ means "there exists" (ie at least one).  Similarly, the restriction on property P `owl:someValuesFrom :A` guarantees that individuals in the restriction class have at least one value on property P from class A.  That is, `owl:someValuesFrom` guarantees that there is some value.

In contrast, the restriction `owl:allValuesFrom` makes no such guarantee. This is because (and as pointed out elsewhere, as well) universal quantification $(\forall)$, which means "for all", is satisfied by *none*.  For example, suppose person `:Joe` has no children. If we then define a property restriction class for the individuals all of whose children are boys, person `:Joe` will satisfy the condition and become a member of the restriction class.  Even though `:Joe` has no children (and hence there is no triple for `:Joe` involving property P), it's valid (logically) to say "all of his children are boys".

So: `owl:allValuesFrom` is satisfied by *none*.

## OWL inferencing / reasoning and subclass

The logic underlying OWL goes beyond what has been described above.

Any class (rdfs:subClassOf) relationship that can be proven to hold, based on the semantics of restrictions, unions, intersections, etc., will be inferred.

The ability in OWL to infer class (rdfs:subClassOf) relationships enables a style of modelling in which subclass relationships are rarely asserted directly. Instead, relationships between classes are described in terms of unions, intersections, complements, and restrictions, and the inferencing engine determines the class structure. Subclass relationships are asserted only in that the members of one class are included in another.  The subclass relationships are substantially inferred from the descriptions of the classes.

This modelling style means that the asserted class hierarchy may look very wide and shallow, with most classes being direct subclasses of `owl:Thing`. Such a hierachy may, at first glance, appear to be uninspired and uninspiring. But this impression can be deceptive. The true class structure implied by the semantics of unions, intersections, complements, restrictions, etc., can be very different and vastly more complex.

For example, suppose classes A, B and C are defined as each being a direct subclass of `owl:Thing`. If class C is defined as being the intersection of classes A and B, then the modeller leaves it to the OWL reasoner to infer that `C subclass A` and `C subclass B`.  

## Reasoning with individuals and classes

From an RDF perspective, inferencing (reasoning) about individuals and about classes are very similar. In both cases, new triples are added to the model based on the triples that were asserted.

From a modelling perspective, the two kinds of reasoning are very different. One of them draws specific conclusions about individuals in a data stream, while the other draws general conclusions about classes of individuals. These two kinds of reasoning are sometimes called *Abox* reasoning (for individuals) and *Tbox* reasoning (for classes). (*Abox* refers to *assertion box*; *Tbox* refers to *terminology box*.)

The utility of reasoning about individuals is clear.  Information (data) is transformed according to a model for use in another context. Data can be transformed and processed according to models and the inference rules associated with RDFS and OWL constructs.

The utility of reasoning about classes is more subtle. It can take place in the absence of any data at all. Class reasoning determines the relationships between classes of individuals.  It determines how data are related in general. In this sense, class reasoning is similar to a compilation of the model. Whereas individual reasoning processes particular data items as input, class reasoning determines general relationships among data and records those relationships with `rdfs:subClassOf`, `rdfs:subPropertyOf`, `rdfs:domain` and `rdfs:range`.  Once these general relationships have been inferred, processing of individual data can be done much more easily.

Reasoning about classes is *general reasoning* about class relationships; reasoning about individuals involves *specific data transformations*.


# Good and Bad Modelling Practices

This section is a summary of Chapter 15 of the `Working Ontologist` book.

## Good naming practices

Here we discuss considerations for choosing the *local* names for our URIs. The local name for a URI is the part of the name after the final slash (for slash URIs) and after the hash (for hash URIs).

Local names can be either *opaque* or *readable*. Opaque local names for URIs are meaningless strings, usually numbers, that are guaranteed to be unique inside that namespace. For example, Wikidata uses opaque local names of the form `Qnnn`, such as `Q41176` ('building'). Readable local names are meaningful for a human reader. For example, DBPedia uses readable local names like `building`.  Both approaches have been used successfully in major ontologies.

Advantages of opaque local names:
* the preferred name of an entity might change as we learn more about our domain; when we use opaque local names, modifying this preferred name is as simple as modifying the entity's label; for example, we could replace triple  
```RDF
wd:Q41176 rdfs:label "building" .
```
with triple
```RDF
wd:Q41176 rdfs:label "structure" .
```
* opaque local names avoid the fallacy of *wishful naming*; that is, if we don't give our URIs a readable name, we won't be tempted to believe that they mean something that the model doesn't express; in the infrastructure of the Web, URIs are actually meaningless; there is no value in pretending otherwise

Disadvantages of opaque local names:
* it is difficult to review a model if you can't read any of the concepts; so students and developers tend to dislike opaque local names; (however, ontology editing tools can mitigate this disadvantage somewhat by displaying concepts based on their label)

Disadvantages of readable local names
* the local part of an HTTP URI is not allowed to have any of a set of special characters; thus, when using readable local names, we are (to some extent) letting a transport protocol (HTTP) have an influence on a human communication form (the name of a concept in a vocabulary)

There is no clear favourite, which is why we see successful examples of both naming approaches in open ontologies today.

### Naming conventions

Conventions have evolved for how to build local names for models in RDFS and OWL.

**Name resources in CamelCase**
In this naming style, multi-word names are written without any spaces but with the first letter of each word (except, perhaps, the first word) written in uppercase.  For example: `rdfs:subClassOf` and `owl:InverseFunctionalProperty`.

**Start class names with capital letters**
For example: `owl:Restriction`, `owl:Class`

**Start property names with lowercase letters**
For example: `rdfs:subClassOf` and `owl:inverseOf`

**Start individual names with capital letters**
For example: `lit:Shakespeare` and `ship:Berengaria`

**Name classes with singular nouns**
For example: `owl:DatatypeProperty` and `owl:SymmetricProperty` and `lit:Playwright`

### Modelling something as a class or individual

It can sometimes be unclear whether it is best to model something as a class or as an individual.  This issue can arise, for example, when considering the potential reuse to which a model (ontology) might be put. There is no hard and fast rule for determining whether something should be modelled as an individual (instance) or a classs. There are, however, some general guidelines that can help organise the decision making process.

The first guideline is based on the observation that classes can be seen as sets of individuals. If something is modelled as a class, then there should at least be a possibility that the class might have instances (individual members). If you cannot imagine what instances would be members of a proposed class, then it is a strong indication that it should not be modelled as a class at all.  That is, if there's no way to imagine what its instances might be, don't model the concept as a class, model it as an individual (instance) instead. If you can imagine instances for the class, it is a good idea to name the class in such a way that the nature of those instances is clear.

The second guideline has to do with the properties that describe the thing to be modelled. If you know (or could know) specific values for those properties then this indicates the concept is better modelled as being an individual (an instance). If you know merely that in general there is some value, this indicates the concept is better modelled as being a class. For example, we know in general that a play has an author, a first performance date, and one or more protagonists, but we know specifically about *The Tempest* that it was written by Shakespeare, was performed in 1611, and has the protagonist Prospero.  In this case, *The Tempest* should be modelled as an individual (instance), and *Play* should be modelled as a class.  And *The Tempest* should be a member of class *Play*.

### Model testing

Once we have assembled a model, how can we test it? Most importantly, we have to be able to determine (by analysing the inferences that the model entails) whether the model maintains consistent answers to possible competency questions from multiple sources. We can also determine test cases for the model and bases of unit tests that can be rerun every time the model evolves. By *model tests* we mean ways we can determine if the model satisfies its intent.

## Common modelling errors

Some modelling practices may be counterproductive for the reuse goals of a semantic model. That is, they do not accomplish the desired goals of sharing information about a structured domain with other stakeholders.  We refer to these counterproductive practices as *antipatterns* --- common pitfalls of beginning modellers that are to be avoided. Here we present several *antipatterns* and outline their drawbacks in terms of the modelling guidelines given above.

### Classism (antipattern)

A common reaction to the difficult distinction between classes and individuals (instances) is simply to *define everything as a class*. Having done so, a beginning modeller might then be tempted to define an object property (an `owl:ObjectProperty`) and use it to relate pairs of classes.

For example, suppose the naive modeller defines all of the following concepts as classes: `:Playwrites`, `:Poets`, `:Shakespeare`, `:Plays`, `:Poems` and `:TheTempest`. Further suppose the modeller defines object property `:wrote`. Finally, suppose the modeller then asserts the following triples
```RDF
:Playwrights :wrote :Plays .
:Poets :wrote :Poems .
:Shakespeare :wrote :Plays .
:Shakespeare :wrote :Poems .
:Shakespeare :wrote :TheTempest .
```

Due to the AAA slogan (anyone can say anything about any topic), we can't say that anything in this set of triples is *wrong*. After all, anyone can assert these triples. But they are flawed in several ways and reflect poor modelling practice.

A) First, the modeller has violated the naming convention that class names are singular.

B) Second, the triple `:Playwrights :wrote :Plays .`, for example, is nonsense.  The resources `:Playwrights` and `:Plays` are classes, ie sets of individuals. Sets don't write plays, individuals do. Further, the triple has no OWL inference semantics. No inferences can be drawn (new triples created) from this assertion, therefore, in terms of OWL semantics, it has no meaning.

C) Third, the triple `:Shakespeare :wrote :TheTempest` is flawed in two respects. 1) `:TheTempest` is modelled as a class. There is no way to imagine what its instances might be. It is a particular play, not a set.  It should have been modelled as an individual. 2) Plays (like The Tempest), are written by people, not by sets. So `:Shakespeare` should be modelled as an individual rather than as a set of individuals (ie as a class).

#### object properties link individuals, not classes

D) Fourth, in general, linking one class to another with an object property (as in all of the triples above) does not support any inferences at all. (Observe that the Protege ontology editor GUI does not offer the user an opportunity to do so.) The only inferences that apply to object properties (like those to do with `rdfs:domain` and `rdfs:range`) assume that the subject and object of the triple are individuals (instances), not classes.   


#### be clear on what you want your model to express

E) Fifth, the naive modeller guilty of classism and of using object properties to link classes, appears to not be clear on what he's trying to express with his model.  That's a key reason why the model is weak.

NOTE: This discussion introduces important and subtle distinctions between the effects of property restrictions `owl:allValuesFrom` and `owl:someValuesFrom` and the construct `owl:equivalenceClass`.

Consider the triple `:Poets :wrote :Poems .`. We have already established that triples like this --- where an object property links two classes (rather than two individuals) --- is nonsense and has no meaning (no inference semantics).  That being the case, how could the intuition behind this naive triple be better expressed?  There are multiple ways.  The only way to choose the correct one is for the modeller to be clear on what it is he wants to express. The modeller must decide precisely what relationship between poets and poems he wants to represent.

**case 1**
For example, the modeller might want to enforce the condition that "If someone is a poet, and he wrote something, then it is a poem."  That is, we might want to ensure we can infer that anything written by a poet is a poem.

Adopting the naming convention of singular nouns for class names, we can enforce this condition and permit such inferences by representing the relationship between poets and poems in this way:
```RDF
:Poet rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty :wrote ;
                        owl:allValuesFrom :Poem ] .
```
This says that Poet is a subclass of the set of individuals who have *only* written poems.  The *only* condition ("for all", universal quantification) that comes from property restriction `owl:allValuesFrom` is what permits us to infer that if a poet has indeed written something, that something is guaranteed to be a poem.  For example, suppose `:Homer` and `:TheIliad` are individuals (poet and poem, respectively). If we had the triples
```RDF
:Homer :wrote :TheIliad .
:Homer rdf:type :Poet .
```
we could infer that
```RDF
:TheIliad rdf:type :Poem .
```

But, notice also that even though this representation of the relationship between poets and poems allows us to infer what we wanted, this representation, as it stands, is an extremely lax definition of poet.  It is lax because the property restriction *only* (`owl:allValuesFrom`) (ie universal quantification) is satisfied by the "none" condition.  In this case this means that individuals who have not written anything also satisfy the condition required for being a poet and hence are included as members of class Poet.  This weakness of this representation of Poet must be recognised and considered before deciding whether to commit to using this representation in one's model.

This representation of the poet/poem relationship has another drawback as well, however.  Achieving the objective of being able to infer that something written by a poet must be a poem can have unintended consequences. Suppose that `:Shakespeare` and `:TheTempest` are now individuals (poet and play, respectively). If we have the triples
```RDF
:Shakespeare :wrote :TheTempest .
:Shakespear rdf:type :Poet .
```
then we end up inferring that
```RDF
:TheTempest rdf:type :Poem .
```
which is an unexpected and unintended conclusion to have reached.

This example illustrates the fact that it is common for poets to write not just (ie only) poems but other things as well. Given this observation, the representation of the relationship between poet and poem given here in *case 1* is probably not the intuition that the modeller was trying to capture with his initial, naive `poets wrote poems` modelling assertion.  

**case 2**
Given the observations derived from *case 1*, the modeller may wish to adapt his conception of what it is to be a poet.  Suppose now that we wish to assert that "If someone is a poet, then they must have written at least one poem."  That is, the condition for being a poet is now that you have written at least one poem.  That is, if someone is a poet we can infer that they have written at least one poem.  That is, being a poet means you have written a poem.

We can represent this adjusted conception of Poet simply by replacing the `owl:allValuesFrom` property restriction with the `owl:someValuesFrom` property restriction:
```RDF
:Poet rdfs:subClassOf [ a owl:Restriction ;
                        owl:onProperty :wrote ;
                        owl:someValuesFrom :Poem ] .
```
This ensures that Poet is a subclass of the set of individuals who have written at least one poem. These individuals may have written things other than poems as well, but they will be guaranteed to have written at least one poem.

If we have a fact such as
```RDF
:Homer rdf:type :Poet .
```
we can infer that he wrote something that is a poem, but we can't necessarily identify what that is.

**case 3**
Given the observations derived from *case 1* and *case 2*, our modeller may wish to consider a third way of representing the concept of Poet and the relationship between poets and poems. The representation in *Case 2* says that being a poet means you have written a poem. Suppose we wish something stronger than this.  Suppose we wish to say: 1) that being a poet means you have written a poem, and 2) if you have written a poem, then you are a poet.

Notice that the inferences permitted by the representation in *case 2* are *uni-directional* only: being a poet means you have written a poem. The inferences permitted by the proposed representation here in *case 3* are *bi-directional*: being a poet means you have written a poem, and having written a poem means you are a poet.

We can realise this ability to make bi-directional inferences by representing (modelling) the relationship between poet and poem using the `owl:equivalentClass` construct in place of the `rdfs:subClassOf` construct, as follows:
```RDF
:Poet rdfs:equivalentClass [ a owl:Restriction ;
                             owl:onProperty :wrote ;
                             owl:someValuesFrom :Poem ] .
```
Now, if we know that
```RDF
:Homer :wrote :TheIliad .
:TheIliad rdf:type :Poem .
```
we can infer that
```RDF
:Homer rdf:type :Poet .
```

**subclass vs equivalenceClass**
Cases 2 and 3, above, illustrate an important and subtle distinction between the effects of constructs `rdfs:subClassOf` and `owl:equivalentClass`.

`rdfs:subClassOf` supports uni-directional inference semantics only

`owl:equivalentClass` supports bi-directional inference semantics. The reason is that this property is symmetric (`owl:SymmetricProperty`), meaning it is its own inverse. So whenever `:x :p :y`, we can infer `:y :p :x`.


### Class punning (antipattern)

In OWL 2.0 it was determined that it is not a logical fallacy to use the same identifier as both a class and an individual (instance).  This is sometimes referred to as *punning*.  (Note that Wikidata permits an entity to be both an 'instance of' one or more classes and a 'subclass of' one or more other classes.)

While this is permitted in OWL 2.0, the authors of the `Working Ontologist`  consider it *poor engineering practice* and recommend not doing this. One reason for this opinion is that the same goal can be achieved, and in a much more precise way, via other means: use of the *Class-Individual Mirror* pattern.

Consider an example involving an endangered species of Eagle and the known individuals of that species. Suppose we call the species `epa:Eagle` and that we know 3 specimens in the wild, each named by the park ranger:
```RDF
epa:Eddie rdf:type epa:Eagle .
epa:Edna rdf:type epa:Eagle .
epa:Elmo rdf:type epa:Eagle .
```
Suppose we want to say more about Eagles. For example, we know that Eagles have wings, and we have heard that the Environmental Protection Agency (EPA) has declared that the Eagle is an endangered species. We can model these things using *punning* as follows:
```RDF
epa:Eagle epa:hasPart epa:Wing .
epa:Eagle rdf:type epa:EndangeredSpecies .
```
This is an example of *punning*. The first triple treats `epa:Eagle` as a class; the second treats `epa:Eagle` as an individual --- an instance of class `epa:EndangeredSpecies`.

But the first triple is also a fallacy.  It is nonsense.  It is another case of the *classism antipattern* where an object property has been used to link two classes rather than two individuals. So this is actually a very poor example of *punning* because teh first triple is so compromised.

When we say "eagles have wings", what we really mean is that each individual eagle has wings. In particular, we mean that `:Eddie` has wings, for example.  Can we conclude from the two triples (ie infer) that `:Eddie` has wings?  No!  The OWL inference model has nothing to say about `:Eddie`; nothing at all.

What we really wanted to say was more like
```RDF
epa:Eagle rdfs:subClassOf [ rdf:type owl:Restriction ;
                            owl:onProperty epa:hasPart ;
                            owl:someValuesFrom epa:Wings ] .
epa:Eagle rdf:type epa:EndangeredSpecies .
```
This is a better example of *punning* because now the first triple at least has meaning (inference semantics). With the first triple we have uni-directional inferencing via `rdfs:subClassOf`: if something is an Eagle, then we can infer that it has parts that are wings. And, as we would wish, the second triple permits no inference about the individual eagles (Eddie, Edna and Elmo). This is what we want because the species being endangered means nothing about an particular individual eagle.

But we still wish to move away from the deprecated use of *punning* altogether. That is, we wish to stop using the single identifier `epa:Eagle` as both a class and an individual (instance). How can we do this while still expressing the two different notions we wish to express?  The solution is to introduce a 3rd concept that can be used to link `epa:Eagle` and `epa:EndangeredSpecies` indirectly.  Define class `epa:Eagle` in terms of this 3rd concept; and then assert that this 3rd concept is an individual (instance) of `epa:EndangeredSpecies`.

As our 3rd concept, suppose we use `epa:EagleSpecies`.  We can then assert
```RDF
epa:EagleSpecies rdf:type epa:EndangeredSpecies .
```
and then relate our 3 individuals to this concept by asserting
```RDF
epa:Eddie epa:hasSpecies epa:EagleSpecies .
epa:Edna epa:hasSpecies epa:EagleSpecies .
epa:Elmo epa:hasSpecies epa:EagleSpecies .
```
We can then define the class `epa:Eagle` as being the class of things that have species Eagle, as follows:
```RDF
epa:Eagle rdf:type owl:Class .
epa:Eagle owl:equivalentClass [ rdf:type owl:Restriction ;
                                owl:onProperty epa:hasSpecies ;
                                owl:hasValue epa:EagleSpecies ] .
```
This definition of class `epa:Eagle` is in fact a simple application of the *Class-Individual Mirror* pattern.

Notice that we still have a class of eagles, and we can say that eagles are endangered. Further, we know quite specifically the relationship that a particular eagle (like Eddie) has to that species.  And we have been able to do all of this without resorting to the *category error* that is *punning*.


### The fallacy of exclusivity (antipattern)

The fallacy of exclusivity is the mistaken assumption that the only candidates for membership in a subclass are those things that are already known to be members of the superclass.

The fallacy of exclusivity
```RDF
:OceanPort rdfs:subClassOf :City .
:OceanPort owl:equivalentClass [ rdf:type owl:Restriction ;
                                 owl:onProperty :connectsTo ;
                                 owl:someValuesFrom :Ocean ] .
```
This model of an ocean port is too open. The restriction class refers to  anything (any individual) that connects to the ocean.  Such things may be cities but not necessarily so. Thus, the restriction admits members which may not be cities. And because an ocean port is declared to be a subclass of `:City`, those members of the restriction class which are not in fact cities will nevertheless be regarded as ocean ports and hence also be inferred to be cities.  For example, suppose we have the triples
```RDF
:France :connectsTo :AtlanticOcean .
:AtlanticOcean rdf:type :Ocean .
```
Given our model of an ocean port (that suffers from the fallacy of exclusivity), these triples lead us to infer that `:France rdf:type :OceanPort .` and furthermore, that `:France rdf:type :City .`.  These inferences are not what we intended by this model. The flaw in the inferencing arises because of the erroneous assumption that only things known to be cities can be ocean ports.  Or, from another perspective, because of the erroneous assumption that only cities can connect to the ocean when, in fact, due to the AAA slogan, anything can connect to the ocean. This fallacy is more a violation of (an ignoring of) the AAA slogan.

The key to solving our modelling problem is to recognise that we want something to become a member of class `:OceanPort` only if it both 'connects to the ocean' and is a city.  In other words, we want to tighten our restriction class. Right now it is too open because, as it stands, it admits non-cities that connect to the ocean as well as cities that connect to the ocean. What we need is an *intersection* of 1) the set of things that 'connect to the ocean' and 2) the set of things that are cities.  That is, we want our class `:OceanPort` to be equivalent to the intersection of our restriction class and the class `:City`.

A revised model that solves the fallacy of exclusivity using just such an intersection of classes is as follows:
```RDF
:OceanPort rdfs:subClassOf :City .
:OceanPort owl:equivalentClass
              [ rdf:type owl:Class ;
                owl:intersectionOf ( :City
                                     [ rdf:type owl:Restriction ;
                                       owl:onProperty :connectsTo ;
                                       owl:someValuesFrom :Ocean ]
                                   )
              ] .
```


### Objectification (antipattern)

Object-oriented (OO) systems are not designed to work in the context of the 3 Semantic Web assumptions: AAA, Open World, and Non-unique Naming. So it's a mistake to try to build a Semantic Web model that has the same meaning and behaviour as an OO system.

The role of a class is different in OO and SW models. In OO models, a class is like a template from which an instance is stamped. Properties are tied to classes. Multiple inheritance is problematic. In SW models, the AAA and Open World assumptions are incompatible with this OO notion of a class.  Properties in SW models exist independently of any class, and because of the AAA slogan, they can be used to describe any individual at all, regardless of the classes to which those individuals might belong. Classes are seen as sets. The OO sense of inheritance (of object structure, data properties and behaviours) doesn't apply. An individual's membership in multiple classes is commonplace and easy to describe using familiar set theory constructs like intersection.  

The intent of an OO model is incompatible with modelling in the SW.

The OO practice of linking properties with classes is contrary to the AAA slogan.  The AAA slogan tells us we can't keep anyone from asserting a property of anything, so we can't enforce the condition that property P can only be specified for individuals of a particular class.

The OO practice of enforcing constraints on properties (eg like a person must have exactly 2 parents) is contrary to the Open World assumption. In the SW, just because we have not asserted a 2nd parent, say, for a given individual, this does not mean that a 2nd parent does not exist. That information might come to light at any time.  The case of having specified one parent only rather than two does not represent a mistake, contradiction or state of error.

Or consider another example on the same point. Suppose more than two parents (say 3) are specified for some person.  In OO this is a mistake. In the SW, not necessarily. The Non-unique Naming assumption means we assume that things (resources, individuals) can have multiple (different) identifiers (URIs) unless told otherwise.  So we accept the possibility that two (or all) of the 3 parent URIs refer to the same person.  The SW model won't detect a contradiction unless it's told that the 3 parents are in fact distinct (using a construct like `owl:differentFrom`, `owl:allDifferent` or `owl:disjointWith`).

An OO model is designed for a very different purpose from a SW model.

Sometimes an OO model cannot be rendered in RDFS/OWL because the requirements of the OO model are simply at odds with the assumptions of modelling in the Semantic Web.


### Creeping conceptualisation (antipattern)

With Semantic Web modelling, all too often the idea of "design for reuse" gets confused with "say everything you can". It is common to mistakenly think that modelling for reuse is best done by anticipating everything that someone might want to use our model for, and thus the more we include (the more classes, the more properties describing our individuals) the better. This is a mistake because the more you put in, the more you *restrict* someone else's ability to *extend* your model instead of just using it as is. Reuse is best done by designing to maximise future combination with other things, not to restrict it.

For example, say we include classes `:ShakespeareanWork` and `:ElizabethanWork` in our model. Having done so, we may well feel tempted to further assert that `:ElizabethanWork` is a subclass of `:Work`, which is a subclass of `:IntangibleWork`. And having included `:IntangibleWork`, we may want to include class `:TangibleWork` as well, and some example individuals of these classes, and some properties of those example individuals, and ... ad infinitum.

It often turns out that knowing when to stop modelling is harder than deciding where to start. As you define one class, you often think immediately of another to which you'd 'naturally' want it to link. This is a natural tendency. Even the best modellers find it difficult to know when to finish.

A relatively easy way to tell if you are going too far in your creation of concepts is to check classes to see if they have properties associated with them, and especially if there are restricted properties. If so, then you are likely saying something useful about them, and they may be included. If you are including data (instances) in your model, then any class that has an instance is likely to be a good class.

But when you see lots of empty classes, especially arranged in a subclass hierarchy, then you are probably creating classes just in case someone might want to do something with them in the future, and that is usually a mistake. The famous acronym KISS (Keep It Simple, Stupid) is well worth keeping in mind when designing Web ontologies.


## Good modelling patterns

### The property transfer pattern

This simple pattern consists of a single triple.  If we have a property P and a property Q and we wish to state that all uses of P should be considered as uses of Q, we can simply assert that
```RDF
:P rdfs:subPropertyOf :Q
```
Now, if have any triple of the form `:x :P :y`, we can infer that `:x :Q :y`.

This use of `rdfs:subPropertyOf` is so pervasive that the authors of the `Working Ontologist` felt it meritted being called out as a pattern in its own right.

However: note that OWL provides `owl:equivalentProperty`, which provides bi-directional inference semantics. So this 'property transfer' pattern may have less value (less need to be employed) if one embraces OWL.


### The class-individual mirror pattern

The *class-individual mirror* pattern is a modelling pattern for relating classes and individuals. It's a pattern for defining a class in terms of an individual. More specifically, for defining a class that represents the set of things that have a particular individual as the *value* of a certain property.  That is, the set of things that relate to a particular individual in a specific way. The pattern therefore involves the `owl:hasValue` property restriction.  The pattern is really just a single `owl:hasValue` restriction on some property, where the value is a particular individual.

The generic form of the *class-individual mirror* pattern is
```RDF
:C owl:equivalentClass [ rdf:type owl:Restriction ;
                         owl:onProperty :P ;
                         owl:hasValue :I ] .
```
where `:C` is a class (set), `:P` is an object property and `:I` is a particular individual (instance).


# Ontology Design Patterns

The book `Semantic Web for the Working Ontologist` introduces many useful ODPs!

Other sources of info on ODPs:

http://ontologydesignpatterns.org/wiki/Main_Page
* initial impression: underwhelmed


# OWL 2.0 subsets

## Introduction

OWL 2.0 includes precise descriptions of four subsets of the OWL language that have been designed for various practical technological reasons. The four subsets are: OWL 2 EL, OWL 2 QL, OWL 2 RL and OWL 2 DL.

Each subset uses the same set of modelling constructs. The distinctions between the subsets of OWL 2.0 are motivated in part by differences in the basic philosophy of why one builds models for the Semantic Web. One philosophy places emphasis on having provable models; the other places emphasis on making executable models.

Provable models
* one motivation for modelling is to be precise about what our models mean
* each construct in OWL is a statement in a formal logic; the particular logical system of OWL DL is called Description Logic
* in certain logics, certain questions cannot be answered by automated means in a finite amount of time; in the study of formal logic, this issue is called *decidability*
* formally, a system is *decidable* if there exists an effective method such that for every formula in the system the method is capable of deciding whether the formula is valid in the system or not; if not, then the system is said to be *undecidable*
* it is easy for a formal system to be undecidable
* even very simple logical systems (basically, any system that can do ordinary integer arithmetic) are undecidable; in fact, it is actually quite challenging to come up with a logical system that can represent anything useful that is also decidable

OWL DL
* OWL DL is based on a particular decidable *Description Logic*
* this means it is possible to design an algorithm that can take as input any model expressed in OWL DL and determine which classes are equivalent to other classes, which classes are subclasses of other classes, and which individuals are members of which classes
* the most commonly used algorithm for this problem is called the *Tableau Algorithm*
* the Tableau Algorithm is guaranteed to find all entailments of a model in OWL DL in a finite (but possibly quite long) time

Executable models
* a different motivation for modelling in the Semantic Web is to form an integrated picture of some sort of domain by federating information from multiple sources
* models are engineered much like software programs; if a model behaves poorly in some situation, an engineer debugs the model until it performs correctly
* decidability is not a primary concern
* provable correctness, as opposed to efficient computation, may not be a key goal
* this opens up the choice of processor for OWL to a much wider range of technologies (than alogrithms like the Tableau Algorithm), including rule systems, datalog engines, databases, and even SPARQL
* this *executable* style of modelling is the primary motivation behind some of the other OWL subsets
* the meaning of an executable model is given by the operations that an inference engine carries out when processing the model


## OWL 2.0 profiles

OWL 2.0 profiles (subsets) were motivated by the observation that certain technologies can process certain subsets of OWL conveniently.

### OWL 2 DL

OWL 2 DL is the largest subset of OWL that retains the desirable feature of being as expressive a language as possible, while still being decidable. It includes all forms of restrictions, combined with all of the RDFS forms.  It is a superset of the following 3 profiles: QL, EL and RL.  It can be processed faithfully by the Tableau Algorithm.

All OWL constructs are allowed but with certain restrictions on their use.

### OWL 2 QL

OWL 2 QL is the subset of OWL 2 designed for applications whereby Semantic Web models are built on top of relational database management systems. Such applications are characterised by fairly simple schema describing the structure of massive amounts of data, and fast responses to queries over that dataset are required. Queries against an OWL 2.0 QL ontology and corresponding data can be rewritten faithfully into SQL, the query language of relational databases.

OWL 2 QL is a subset of OWL 2.0 restricted to be compatible with SQL database queries.  OWL 2 QL is designed to be the front end of database query languages.

### OWL 2 EL

Large ontologies are difficult to process using an unconstrained Tableau Algorithm. The OWL 2 EL profile was designed for such large ontologies. It trades-off some expressive power in return for being able to exploit certain known optimisations for querying large structures efficiently.  It allows  `owl:someValuesFrom` property restriction classes but not `owl:allValuesFrom` or `owl:hasValue` property restriction classes.  This profile is restricted enough that fast algorithms are known for processing very large ontologies.

OWL 2 EL is a subset of OWL 2.0 restricted to improve computational efficiency.

### OWL 2 RL

Many OWL processors work by using rules-based technology to define OWL processing.  (RDFox, for example, uses datalog rules and reasoning, even for OWL axioms, which are converted to datalog rules.)  Rules processors have been around for ages, in the form of systems like Prolog and Business Rules engines.  OWL 2 RL defines the subset of OWL that can be faithfully processed by such rule systems.

OWL 2 RL is a subset of OWL 2.0 restricted to be compatible with Rules processors.

### Final comments

Any model in any profile can be interpreted as a model in any other profile --- subject to the restrictions of that profile. All of the profiles are interoperable at the RDF level.


## Rules

Even OWL 2.0 has some limits to its expressivity. Some of othese limitations are best addressed using *rules*. A rules language for the Web has been developed in the form of the RIF (Rules Interchange Format). There are also many proprietary rules processors.

Rules-based systems started in the days of Expert Systems.

### Uncertainty

There are several approaches to reasoning with uncertainty in rules. Many of these have considerable research and practical examples behind them, making uncertainty in rules a relatively well-understood issue.



# Ontology Engineering

You use whatever OWL constructs are available to express the semantics of the things you care creating. The ontology gives meaning to the data, and adding more meaning to the ontology adds more meaning to the data.

When you create an ontology you are creating a vocabulary of terms to define the subject matter of interest (a domain). But it's much more than a vocabulary because you also carefully define the meaning of the terms using the available OWL constructs.

After you create your vocabulary of classes and properties, you use it to create your data (assertions about individuals):
* classes arranged in class hierarchy
* properties arranged in property hierarchy
* designating properties as being either object properties (relating an individual to another individual) or data properties (relating an individual to a literal value)
* data properties are used to describe the attributes of individuals

## Individuals, classes and properties

Individuals
* a specific thing (person, place or thing)
* `owl:Thing` and `owl:NamedIndividual` say something is an individual
* `rdf:type` says what class an individual is a member of
* `:Joe rdf:type owl:Thing`, `:Joe rdf:type owl:NamedIndividual`
* `:Joe rdfs:label "Joe Bloggs"`

Classes
* a kind of thing
* the set of all things of that specific kind
* `owl:Class` says something is a class: `:B rdf:type owl:Class`
* a subclass is a more specific kind of thing than its superclass
* `rdfs:subClassOf` says a class is a subclass of another class
* every member of the subclass is also a member of the superclass
* `:A rdfs:subClassOf :B`

Properties
* a property relates individuals to each other or to literals
* there are 3 main kinds of properties
* object property: relates an individual to another individual  
  - `:Joe :hasFriend :Jill`
* data property: relates an individual to a literal value
  - `:Joe :hasAge 35`
* annotation property: like object and data properties but without any inference semantics
  - `:Joe rdfs:label "Joe Bloggs"`
* a property is said to be a *subproperty* of another property if it represents a more specific kind of relationship than it parent property
* `:employedBy rdfs:subPropertyOf :worksFor`
* the main kinds of assertions relating to properties are:
  - create an object property and specify its inverse
  - use an object property in an assertion (ie assert that one individual has a certain relationship with another individual)
  - create a data property: `:hasAge rdf:type owl:DatatypeProperty`
  - use a data property in an assertion: `:Joe :hasAge 33`





asdf
