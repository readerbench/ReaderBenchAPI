import json
from typing import Dict, List, Tuple

from flask import Flask, jsonify, request
from rb.complexity.complexity_index import compute_indices
from rb.complexity.index_category import IndexCategory
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.core.word import Word
from rb.processings.keywords.keywords_extractor import extract_keywords
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import VECTOR_MODELS, get_default_model
from rb.utils.utils import str_to_lang

app = Flask(__name__)


def keywordsOption():
    return ""

def transform_for_visualization(keywords: List[Tuple[int, Word]], lang: Lang) -> Dict:

    vector_model: VectorModel = get_default_model(lang)
    edge_list, node_list = [], []

    for i, kw1 in enumerate(keywords):
        for j, kw2 in enumerate(keywords):
            sim = vector_model.similarity(vector_model.get_vector(kw1[1]), vector_model.get_vector(kw2[1]))
            if i != j and sim >= 0.3:
                edge_list.append({
                    "edgeType": "SemanticDistance",
                    "score": str(max(sim, 0)),
                    "sourceUri": kw1[1],
                    "targetUri": kw2[1]
                })

    for kw in keywords:
        node_list.append({
            "type": "Word",
            "uri": kw[1],
            "displayName": kw[1],
            "active": True,
            "degree": str(max(0, float(kw[0])))
        })

    return {
        "data": {
            "edgeList": edge_list,
            "nodeList": node_list
        },
        "success": True,
        "errorMsg": ""
    }


def keywordsPost():
    """TODO, not working"""
    params = json.loads(request.get_data())
    posTagging = params.get('pos-tagging')
    bigrams = params.get('bigrams')
    text = params.get('text')
    languageString = params.get('language')
    lang = str_to_lang(languageString)
    threshold = params.get('threshold')

    # if lang is Lang.RO:
    #     vector_model = VECTOR_MODELS[lang][CorporaEnum.README][VectorModelType.WORD2VEC](
    #         name=CorporaEnum.README.value, lang=lang)
    # elif lang is Lang.EN:
    #     vector_model = VECTOR_MODELS[lang][CorporaEnum.COCA][VectorModelType.WORD2VEC](
    #         name=CorporaEnum.COCA.value, lang=lang)
    # elif lang is Lang.ES:
    #     vector_model = VECTOR_MODELS[lang][CorporaEnum.JOSE_ANTONIO][VectorModelType.WORD2VEC](
    #         name=CorporaEnum.JOSE_ANTONIO.value, lang=lang)

    # lsa = params.get('lsa')
    # lda = params.get('lda')
    # w2v = params.get('w2v')
    # threshold = params.get('threshold')

    # textElement = Document(lang=lang, text=text, vector_model=vector_model)
    # print(textElement.keywords)

    keywords = extract_keywords(text=text, lang=lang, threshold=threshold)
    return jsonify(transform_for_visualization(keywords=keywords, lang=lang))
