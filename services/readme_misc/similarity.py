from typing import Union, List, Dict, Any

from rb import Word, Lang
from rb.similarity.wordnet import get_synsets, get_synset_hypernyms, lang_dict


def get_hypernymes_grouped_by_synset(word: Union[str, Word], lang: Lang = None, pos: str = None) -> Dict[Any, List[str]]:    
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