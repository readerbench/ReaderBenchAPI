from typing import Union, List, Dict, Any

from rb import Word, Lang
from rb.parser.spacy_parser import SpacyParser
from rb.similarity.wordnet import get_synsets, get_synset_hypernyms

import rowordnet as rwn
import spacy

nlp = spacy.load("ro_core_news_lg")
wn = rwn.RoWordNet()

lang_dict = {
    Lang.EN: 'eng',
    Lang.NL: 'nld',
    Lang.FR: 'fra',
    Lang.RO: 'ron',
    Lang.IT: 'ita'
}


def get_hypernymes_grouped_by_synset(word: Union[str, Word], lang: Lang = None, pos: str = None) -> Dict[Any, list]:    
    if lang == Lang.RO:
        # lemmatize
        doc = nlp(word)
        if len(doc) > 0:
            word = doc[0].lemma_
        # using rowordnet
        result = dict()
        synsets = wn.synsets(literal=word, strict=True)
        for synset_id in synsets:
            parent = wn.synset(synset_id)
            print(parent)
            if result.get(parent.domain, None):
                result[parent.domain]['hypernyms'].update(parent.literals)
            else: 
                result[parent.domain] = dict()
                result[parent.domain]['hypernyms'] = set(parent.literals)
                result[parent.domain]['definition'] = parent.definition
        
        for key in result: 
            # hypernym set to list
            # remove _ from multiple words
            result[key]['hypernyms'] = [hypernym.replace("_", " ") for hypernym in result[key]['hypernyms']]
        print(result)
        return result

    if isinstance(word, Word):
        pos = word.pos.to_wordnet()
        lang = word.lang
        word = word.lemma
    if lang not in lang_dict:
        return []
    result = dict()

    # using plain wordnet
    synsets = get_synsets(word, pos=pos, lang=lang_dict[lang])
    for synset in synsets:
        for parent in get_synset_hypernyms(synset):
            lang_lemmas = []
            lemmas = parent.lemma_names(lang=lang_dict[lang])
            for lemma in lemmas:
                lang_lemmas.append(lemma)
            if lang_lemmas:
                result[lang_lemmas[0]] = dict()
                result[lang_lemmas[0]]['hypernyms'] = lang_lemmas
                result[lang_lemmas[0]]['definition'] = parent.definition()
    return result


result = get_hypernymes_grouped_by_synset("corp", Lang.RO)
print(result)