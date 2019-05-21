from flask import Flask, request
import json
from rb.core.lang import Lang
from rb_api.amoc.comprehension_model_service import ComprehensionModelService
from rb.similarity.word2vec import Word2Vec
from rb_api.dto.amoc.amoc_response import AmocResponse

app = Flask(__name__)


def amocOption():
    return ""


def amocPost():
    params = json.loads(request.get_data())
    text = params.get("text")
    semantic_model = params.get("semanticModel")
    min_activation_threshold = params.get("minActivationThreshold")
    max_active_concepts = params.get("maxActiveConcepts")
    max_semantic_expand = params.get("maxSemanticExpand")

    semantic_models = [Word2Vec(semantic_model, Lang.EN)]
    cms = ComprehensionModelService(semantic_models, Lang.EN,
                                    min_activation_threshold, max_active_concepts, max_semantic_expand)

    result = cms.run(text)
    amoc_response = AmocResponse(result, "", True)

    return amoc_response