from flask import Flask, request
import json
from rb.core.lang import Lang
from rb.utils.utils import str_to_lang
from rb_api.amoc.comprehension_model_service import ComprehensionModelService
from rb.similarity.vector_model import VectorModelType
from rb.similarity.word2vec import Word2Vec
from rb.similarity.lsa import LSA
from rb.similarity.lda import LDA
from rb_api.dto.amoc.amoc_response import AmocResponse
import rb_api.cache.cache as cache

app = Flask(__name__)


def amocOption():
    return ""


def amocPost():
    params = json.loads(request.get_data())
    text = params.get("text")
    semantic_model = params.get("semanticModel")
    min_activation_threshold = float(params.get("minActivationThreshold"))
    max_active_concepts = int(params.get("maxActiveConcepts"))
    max_semantic_expand = int(params.get("maxSemanticExpand"))
    languageString = params.get('language')
    lang = str_to_lang(languageString)

    w2v = cache.get_model(VectorModelType.WORD2VEC, semantic_model, lang)
    lda = cache.get_model(VectorModelType.LDA, semantic_model, lang)
    lsa = cache.get_model(VectorModelType.LSA, semantic_model, lang)
    semantic_models = [w2v, lda, lsa]
    cms = ComprehensionModelService(semantic_models, lang,
                                    min_activation_threshold, max_active_concepts, max_semantic_expand)

    result = cms.run(text)
    amoc_response = AmocResponse(result, "", True)

    return amoc_response.toJSON()