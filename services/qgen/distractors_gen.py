# !pip install sense2vec==1.0.2
# !wget https://github.com/explosion/sense2vec/releases/download/v1.0.0/s2v_reddit_2015_md.tar.gz
# !tar -xvf  s2v_reddit_2015_md.tar.gz
# !pip install SPARQLWrapper

import xml.etree.ElementTree as ET

import nltk
import requests
from SPARQLWrapper import JSON, SPARQLWrapper

from services.qgen.utils import get_fitness_loss_no_finetune

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

import re

from nltk.corpus import wordnet as wn
from sense2vec import Sense2Vec

s2v = Sense2Vec().from_disk('models/s2v_reddit_2019_lg')
from collections import OrderedDict


def get_distractors_wordnet(word):
    syn = wn.synsets(word,'n')[0]
    distractors=[]
    word= word.lower()
    orig_word = word
    if len(word.split())>0:
        word = word.replace(" ","_")
    hypernym = syn.hypernyms()
    if len(hypernym) == 0: 
        return distractors
    for item in hypernym[0].hyponyms():
        name = item.lemmas()[0].name()
        #print ("name ",name, " word",orig_word)
        if name == orig_word:
            continue
        name = name.replace("_"," ")
        name = " ".join(w.capitalize() for w in name.split())
        if name is not None and name not in distractors:
            distractors.append(name)
    return distractors

def get_distractors_conceptnet(word):
    word = word.lower()
    original_word= word
    if (len(word.split())>0):
        word = word.replace(" ","_")
    distractor_list = [] 
    url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5"%(word,word)
    obj = requests.get(url).json()

    for edge in obj['edges']:
        link = edge['end']['term'] 

        url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10"%(link,link)
        obj2 = requests.get(url2).json()
        for edge in obj2['edges']:
            word2 = edge['start']['label']
            if word2 not in distractor_list and original_word.lower() not in word2.lower():
                distractor_list.append(word2)
                   
    return distractor_list

def get_distractors_sense2vec(word):
    output = []
    word = word.lower()
    word = word.replace(" ", "_")

    sense = s2v.get_best_sense(word)
    most_similar = s2v.most_similar(sense, n=10)

    # print ("most_similar ",most_similar)

    for each_word in most_similar:
        append_word = each_word[0].split("|")[0].replace("_", " ").lower()
        if append_word.lower() != word:
            output.append(append_word.title())

    out = list(OrderedDict.fromkeys(output))
    return out

def locate_dbpedia_uri(word, sentence):
    service_url = "https://lookup.dbpedia.org/api/search"
    headers = {"Accept": "application/xml", "Accept-Language": "en"}
    params = {"query": word, "maxResults": 10}
    response = requests.get(service_url, headers=headers, params=params)

    root = ET.fromstring(response.content)
    results = root.findall("Result")

    word_uri = [r.find("URI").text for r in results]
    words = [w.split('/')[-1].replace('_', ' ') for w in word_uri]

    uri_losses = get_fitness_loss_no_finetune(sentence, words, word)
    best_uris = [(u, l) for u, l in zip(word_uri, uri_losses)]
    best_uris = sorted(best_uris, key=lambda x: x[1])
    print(best_uris[0])
    return best_uris[0][0]

def get_distractors_dbpedia2(input_uri):
    endpoint_url = "http://dbpedia.org/sparql"

    # Define the SPARQL query
    query = """
    SELECT ?entity (COUNT (?commonType) AS ?commonTypeCount) ?label WHERE {{
    <{0}> rdf:type ?commonType .
    FILTER(STRSTARTS(STR(?commonType), "http://dbpedia.org/class/yago") || STRSTARTS(STR(?commonType), "http://dbpedia.org/ontology"))	
    ?entity rdf:type ?commonType .
    FILTER (?entity != <{0}>)
    ?entity rdfs:label ?label .
    FILTER(LANG(?label) = 'en')
    }} GROUP BY ?entity ?label ORDER BY DESC(?commonTypeCount)
    """.format(input_uri)

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    resources = [result["entity"]["value"] for result in results["results"]["bindings"]]
    type_counts = [result["commonTypeCount"]["value"] for result in results["results"]["bindings"]]
    labels = [result["label"]["value"] for result in results["results"]["bindings"]]

    return labels[:10]

def get_distractors_dbpedia3(input_uri):
    endpoint_url = "http://dbpedia.org/sparql"

    # Define the SPARQL query
    query = """
    SELECT ?entity ?label WHERE {{
    <{0}> dbo:wikiPageWikiLink ?entity.
    ?entity rdfs:label ?label .
    FILTER(LANG(?label) = 'en')
    }}
    """.format(input_uri)

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    resources = [result["entity"]["value"] for result in results["results"]["bindings"]]
    labels = [result["label"]["value"] for result in results["results"]["bindings"]]

    return labels

def get_distractors_dbpedia4(input_uri):
    endpoint_url = "http://dbpedia.org/sparql"

    # Define the SPARQL query
    query = """
    SELECT ?entity ?label WHERE {{
    <{0}> gold:hypernym ?hypernym.
    ?entity gold:hypernym ?hypernym.
    ?entity rdfs:label ?label .
    FILTER(LANG(?label) = 'en')
    }}
    """.format(input_uri)

    sparql = SPARQLWrapper(endpoint_url)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    results = sparql.query().convert()

    resources = [result["entity"]["value"] for result in results["results"]["bindings"]]
    labels = [result["label"]["value"] for result in results["results"]["bindings"]]

    return labels[:10]

def get_distractors_context(predicate, triplets):
    return [t['tail'] for t in triplets if t['type'] ==  predicate]
