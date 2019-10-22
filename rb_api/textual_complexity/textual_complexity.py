from flask import Flask, request
import json
import sys
from rb.core.text_element import TextElement
from rb.core.lang import Lang
from rb.utils.utils import str_to_lang
from rb.complexity.index_category import IndexCategory
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.similarity.vector_model import VectorModelType, CorporaEnum, VectorModel
from rb.similarity.vector_model_instance import VECTOR_MODELS
from rb.cna.cna_graph import CnaGraph
from rb_api.dto.textual_complexity.complexity_index_dto import ComplexityIndexDTO
from rb_api.dto.textual_complexity.textual_complexity_data_dto import TextualComplexityDataDTO
from rb_api.dto.textual_complexity.textual_complexity_response import TextualComplexityResponse
from rb_api.dto.textual_complexity.complexity_indices_dto import ComplexityIndicesDTO

app = Flask(__name__)


def textualComplexityOption():
    return ""


def textualComplexityPost():
    params = json.loads(request.get_data())
    posTagging = params.get('pos-tagging')
    dialogism = params.get('dialogism')
    bigrams = params.get('bigrams')
    text = params.get('text')
    languageString = params.get('language')
    lang = str_to_lang(languageString)
    lsa = params.get('lsa')
    lda = params.get('lda')
    w2v = params.get('w2v')
    threshold = params.get('threshold')

    if lang is Lang.RO:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.README][VectorModelType.WORD2VEC](
            name=CorporaEnum.README.value, lang=lang)
    elif lang is Lang.EN:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.COCA][VectorModelType.WORD2VEC](
            name=CorporaEnum.COCA.value, lang=lang)

    document = Document(lang=lang, text=text)
    cna_graph = CnaGraph(doc=document, models=[vector_model])
    compute_indices(doc=document, cna_graph=cna_graph)

    categoriesList = []
    complexityIndices = {}
    for key, value in document.indices.items():

        categoryName = key.category.name
        if (categoryName not in categoriesList):
            categoriesList.append(categoryName)

        complexityIndexDTO = ComplexityIndexDTO(key.abbr + " (document)", value)
        # complexityIndex[categoryName] = complexityIndexDTO
        if (not categoryName in complexityIndices):
            complexityIndices[categoryName] = []
        complexityIndices[categoryName].append(complexityIndexDTO)

    for paragraph in document.components:
        for key, value in paragraph.indices.items():
            categoryName = key.category.name
            if (categoryName not in categoriesList):
                categoriesList.append(categoryName)

            complexityIndexDTO = ComplexityIndexDTO(key.abbr + " (paragraph)", value)
            # complexityIndex[categoryName] = complexityIndexDTO
            if (not categoryName in complexityIndices):
                complexityIndices[categoryName] = []
            complexityIndices[categoryName].append(complexityIndexDTO)

    for paragraph in document.components:
        for sentence in paragraph.components:
            for key, value in sentence.indices.items():
                categoryName = key.category.name
                if (categoryName not in categoriesList):
                    categoriesList.append(categoryName)

                complexityIndexDTO = ComplexityIndexDTO(key.abbr + " (sentence)", value)
                # complexityIndex[categoryName] = complexityIndexDTO
                if (not categoryName in complexityIndices):
                    complexityIndices[categoryName] = []
                complexityIndices[categoryName].append(complexityIndexDTO)

    # iterate through complexity index array
    complexityIndicesResponse = []
    for keyCategory, indicesCategory in complexityIndices.items():
        valences = []
        for complexityIndex in indicesCategory:
            valences.append(complexityIndex)
        complexityIndicesDTO = ComplexityIndicesDTO(keyCategory, valences)
        complexityIndicesResponse.append(complexityIndicesDTO)

    textualComplexityDataDTO = TextualComplexityDataDTO(
        languageString, categoriesList, complexityIndicesResponse)

    textualComplexityResponse = TextualComplexityResponse(
        textualComplexityDataDTO, "", True)
    jsonString = textualComplexityResponse.toJSON()
    # print(textualComplexityResponse)
    # jsonString = json.dumps(textualComplexityResponse, default=TextualComplexityResponse.dumper, indent=2)

    return jsonString
