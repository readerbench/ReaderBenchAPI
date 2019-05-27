from flask import Flask, request
import json
from rb.core.text_element import TextElement
from rb.core.lang import Lang
from rb.complexity.index_category import IndexCategory
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.word import Word
from rb_api.dto.textual_complexity.complexity_index_dto import ComplexityIndexDTO
from rb_api.dto.textual_complexity.textual_complexity_data_dto import TextualComplexityDataDTO
from rb_api.dto.textual_complexity.textual_complexity_response import TextualComplexityResponse
from rb_api.dto.textual_complexity.complexity_indices_dto import ComplexityIndicesDTO
from rb.similarity.vector_model import VectorModel, VectorModelType

app = Flask(__name__)


def textSimilarityOption():
    return ""


def textSimilarityPost():
    params = json.loads(request.get_data())
    corpus = params.get('corpus')
    languageString = params.get('language').upper()
    model = params.get('model')
    text1 = params.get('text1')
    text2 = params.get('text2')
    lang = Lang[languageString]

    document1 = Document(lang, text1)
    document2 = Document(lang, text2)
    # complexity = ComplexityIndex(lang, IndexCategory.SYNTAX, "syntax")
    vectorModel = VectorModel(VectorModelType.WORD2VEC, '', lang)
    similarityScore = vectorModel.similarity(document1, document2)
    
    word1 = Word(lang, 'dog')
    word2 = Word(lang, 'cat')
    similarityScore = vectorModel.similarity(word1, word2)
    print(similarityScore)

    # textualComplexityResponse = TextualComplexityResponse(textualComplexityDataDTO, "", True)
    # jsonString = textualComplexityResponse.toJSON()
    jsonString = ''

    return jsonString
