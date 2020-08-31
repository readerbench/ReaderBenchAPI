from flask import Flask, request, jsonify
import json
try:
    from rb.core.text_element import TextElement
    from rb.core.lang import Lang
    from rb.complexity.index_category import IndexCategory
    from rb.complexity.complexity_index import compute_indices
    from rb.core.document import Document
    from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
    from rb.similarity.vector_model_factory import VECTOR_MODELS
    from rb.utils.utils import str_to_lang
    from rb.processings.keywords.keywords_extractor import KeywordExtractor
except:
    import sys
    sys.path.insert(0, '../readerbenchpy')
    from rb.core.text_element import TextElement
    from rb.core.lang import Lang
    from rb.complexity.index_category import IndexCategory
    from rb.complexity.complexity_index import compute_indices
    from rb.core.document import Document
    from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
    from rb.similarity.vector_model_instance import VECTOR_MODELS
    from rb.utils.utils import str_to_lang
    from rb.processings.keywords.keywords_extractor import KeywordExtractor
app = Flask(__name__)


def keywordsOption():
    return ""


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

    keywords_extractor = KeywordExtractor()
    keywords = keywords_extractor.extract_keywords(text=text, lang=lang, threshold=threshold)
    return jsonify(keywords_extractor.transform_for_visualization(keywords, lang=lang))
