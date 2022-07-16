# Wikidata - query it with SPARQL {ignore=true}

**Table of Contents**
[TOC]

# Wikidata info sources

https://www.wikidata.org/wiki/Wikidata:Main_Page

https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial

https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service

https://www.wikidata.org/wiki/Wikidata:SPARQL_query_service/queries/examples

https://www.wikidata.org/wiki/Wikidata:Data_access
* a starting point for obtaining data from Wikidata
* wikidata sparql endpoint: https://query.wikidata.org/sparql

https://www.wikidata.org/wiki/Wikidata:WikiProject_Ontology/Modelling
* class (Q16889133):
  - the class of all classes; so all classes are an instance of 'class'
  - things that conceptually group together similar items
  - the items in a class are known as its *instances*
  - they are related to the class via *instance of* (P31)
  - classes don't need to have instances (1 class : * instances)
  - every item that is value of *instance of* (P31) is a class
  - :Qitem :P31 :Qclass
  - :Q567 :P31 :Q5  (Angela Merkel instance_of human)
* subclass of (P279):
  - classes are related to more-general classes using *subclass of* (P279)
* entity (Q25120):
  - this is the class of all items, so all items are an instance of 'entity', and all classes are subclasses of 'entity'
* items:
  - every item should be an instance of one or more classes
  - If an item is an instance of a class then it is also an instance of any more-general classes
  - every item that is a *subclass of* of a class is itself a class; this is why it's not necessary for most classes to explicitly state that they are an *instance of* class 'class' (Q16889133)
* classes can be instances of other classes

classes (and subclasses) *group* things; instances *are* things:
* items can be both instances of and subclasses of at the same time
* an item should not be both an instance of and a subclass of the same class
* the instances of a class should not mix together groups of things and the things themselves
* the subclasses of a class should not mix together groups of things and the things themselves
* consider Q5 (human):
  - Q5 is all these: an instance, a subclass and part of something
  - instance of: 'Homo sapiens'
  - subclass of: 'person' (Q215627)
  - part of: 'humanity'
* consider Q23444 (white):
  - instance of: 'color'    (white)
  - subclass of: 'light'    (white light)
  - the fact that the two classes, 'color' and 'light', are so different from one another helps one see how it is that an item, like Q23444 (white), can be both an instance of one class and a subclass of another class
  - so, item Qx can be both 'instance of' class A and 'subclass of' class B
  - the semantics of the item, Qx, adjusts (somewhat) depending on which target class, A or B, is the perspective from which the item is being considered
  - there is a context switch, so it's ok to consider the concept one way in context A and another way in context B.  For example, when regarded as a 'color', 'white' is an instance; when regarded as 'light', 'white' really means 'white light', so it's something very different from the 'color' white; we're now talking about a category of 'light', a subclass of 'light'
  - so, one can interpret examples like Q23444 (white) as *bad form* or *weak domain modelling* or *deprecated*, because a single term, 'white', is being used to represent two very different things: a color, and a category of light.  Indeed, one could point to a solution strategy: add a node to the KG for 'white light', and make it a subclass of 'light'; amend 'white' so it's just instance of 'color'
  * so, *we deprecate the pattern* of defining an item that is simultaneously an *instance of* one class and a *subclass of* another class  
* consider Q1075 (color):
  - 'primary color' subclass_of color
  - 'white' instance_of color
  - 'white' subclass_of light

https://www.wikidata.org/wiki/Wikidata:Item_classification
* item classification is done via non-domain specific properties:
  - subclass of (P279)  (aka rdfs:subClassOf)
  - instance of (P31)   (aka rdf:type)


**Lists of properties**
List of properties (all in one table)
https://www.wikidata.org/wiki/Wikidata:List_of_properties/all_in_one_table
* there's warning that this page is outdated, but it's still very helpful as a starting point

https://www.wikidata.org/wiki/Wikidata:Database_reports/List_of_properties/all
* this page has no outdated warning; it looks safe!

**Wikidata Query Service**
https://query.wikidata.org/
very helpful!!!


# Wikidata ontology info

Display a wikidata item:
* get item UID for the item whose page you want to see
* eg say,  Q7378  (elephant)
* open browser and enter URL
https://www.wikidata.org/wiki/Q7378

Key Wikidata properties by UID:
* P279 == 'subclass of' == rdfs:subClassOf
* P31 == 'instance of' == rdf:type
* P361 == 'part of'
* P1552 == 'has quality'

Key Wikidata properties by URI:
* rdfs:label                Label
* schema:description        Description
* schema:version
* schema:dateModified


# Example Wikidata queries

Get the subsumption chain (the chain of superclasses) for a given Wikidata entity UID:
```
SELECT ?class ?classLabel ?superclass ?superclassLabel
WHERE
{
    wd:Q7378 wdt:P279 ?class .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
```


Get the parent entity and grandparent entity:
```
SELECT ?class ?classLabel ?superclass ?superclassLabel
WHERE
{
    wd:Q7378 wdt:P279 ?class .
    ?class wdt:P279 ?superclass .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
```

https://stackoverflow.com/questions/66191123/how-to-get-all-superclasses-of-a-wikidata-entity-with-sparql

Get all superclasses of an entity:
```
SELECT ?class ?classLabel ?superclass ?superclassLabel
WHERE
{
    wd:Q125977 wdt:P279* ?class .
    ?class wdt:P279 ?superclass .
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
```

List all identifiers of a given property
```
SELECT ?item ?itemLabel ?id
WHERE {
  ?item wdt:P4466 ?id
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
```

Show all items with a property
```
# Sample to query all values of a property
# Property talk pages on Wikidata include basic queries adapted to each property
SELECT
  ?item ?itemLabel
  ?value ?valueLabel
# valueLabel is only useful for properties with item-datatype
WHERE
{
  ?item wdt:P1800 ?value
  # change P1800 to another property        
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
# remove or change limit for more results
LIMIT 10
```

Get item label (in a specific language) for a given item UID
```
SELECT  ?label
WHERE {
        wd:Q146190 rdfs:label ?label .
        FILTER (langMatches( lang(?label), "EN" ) )
      }
LIMIT 3
```

Get item UID(s) for a given item label:
```
SELECT  ?item ?itemLabel
WHERE {
        ?item rdfs:label "elephant"@en .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
      }
LIMIT 10
```

Find the name of the property that holds the 'Description' of a Wikidata entity (UID):
Q7378 has label 'elephant'
```
SELECT  ?prop
WHERE {
        wd:Q7378 ?prop "trunk-bearing large mammal"@en .
        SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
      }
LIMIT 10
```

Confirm 'schema:description' works as we expect:
```
SELECT  ?desc
WHERE {
        wd:Q7378 schema:description ?desc .
      }
LIMIT 10
```
Outcome: many matches (one per language)
Note:
* defining a PREFIX such as `schema: <http://www.schema.org/>` leads to 'no matches found'; so it's best NOT to define prefixes for the default prefixes that Wikidata recognises automatically

Get just the English description of an entity:
```
SELECT  ?desc
WHERE {
        wd:Q7378 schema:description ?desc .
        FILTER (lang(?desc) = 'en')
      }
LIMIT 10
```

Get items for a given Description:
```
SELECT ?item ?itemLabel
WHERE {
    ?item schema:description "luggage"@en .
    SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en' . }
}
LIMIT 10
```

Get items for a given Also Known As token:
```
SELECT ?item ?itemLabel
WHERE {
    ?item skos:altLabel "luggage"@en .
    SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en' . }
}
LIMIT 10
```

Find the subclasses of a given class:
* P279 == 'subclass of'
* Q7377 == mammal
```
SELECT ?item ?itemLabel
WHERE {
    ?item wdt:P279 wd:Q7377 .
    SERVICE wikibase:label { bd:serviceParam wikibase:language '[AUTO_LANGUAGE],en' . }
}
LIMIT 50
```


# Trouble Shooting

## HTTPError: Forbidden

https://en.wikipedia.org/wiki/HTTP_403
* The HTTP 403 is a HTTP status code meaning access to the requested resource is forbidden. The server understood the request, but will not fulfill it.
* HTTP 403 is returned when the client is not permitted access to the resource despite providing authentication
* Error 403: "The server understood the request, but is refusing to authorize it.", RFC 7231















asdf
