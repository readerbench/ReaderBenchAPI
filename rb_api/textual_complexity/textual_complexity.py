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
from rb.similarity.vector_model_factory import VECTOR_MODELS
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
    text = params.get('text')
    languageString = params.get('language')
    lang = str_to_lang(languageString)
    # lsa = params.get('lsa')
    # lda = params.get('lda')
    # w2v = params.get('w2v')

    if lang is Lang.RO:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.README][VectorModelType.WORD2VEC](
            name=CorporaEnum.README.value, lang=lang)
    elif lang is Lang.EN:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.COCA][VectorModelType.WORD2VEC](
            name=CorporaEnum.COCA.value, lang=lang)
    elif lang is Lang.ES:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.JOSE_ANTONIO][VectorModelType.WORD2VEC](
            name=CorporaEnum.JOSE_ANTONIO.value, lang=lang)
    elif lang is Lang.FR:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.LE_MONDE][VectorModelType.WORD2VEC](
            name=CorporaEnum.LE_MONDE.value, lang=lang)
    elif lang is Lang.RU:
        vector_model = VECTOR_MODELS[lang][CorporaEnum.RNC_WIKIPEDIA][VectorModelType.WORD2VEC](
            name=CorporaEnum.RNC_WIKIPEDIA.value, lang=lang)

    document = Document(lang=lang, text=text)
    cna_graph = CnaGraph(docs=document, models=[vector_model])
    compute_indices(doc=document, cna_graph=cna_graph)

    categoriesList = []
    complexityIndices = {}
    for key, value in document.indices.items():
        categoryName = key.category.name
        if (categoryName not in categoriesList):
            categoriesList.append(categoryName)

        complexityIndexDTO = ComplexityIndexDTO(
            repr(key), float(value), type="document")
        # complexityIndex[categoryName] = complexityIndexDTO
        if categoryName not in complexityIndices:
            complexityIndices[categoryName] = []
        complexityIndices[categoryName].append(complexityIndexDTO)

    for paragraph_id, paragraph in enumerate(document.components):
        for key, value in paragraph.indices.items():
            categoryName = key.category.name
            if (categoryName not in categoriesList):
                categoriesList.append(categoryName)

            complexityIndexDTO = ComplexityIndexDTO(repr(key), float(value), 
                                                    type="paragraph", 
                                                    paragraph_index= paragraph_id)
            # complexityIndex[categoryName] = complexityIndexDTO
            if categoryName not in complexityIndices:
                complexityIndices[categoryName] = []
            complexityIndices[categoryName].append(complexityIndexDTO)

    for paragraph_id, paragraph in enumerate(document.components):
        for sentence_id, sentence in enumerate(paragraph.components):
            for key, value in sentence.indices.items():
                categoryName = key.category.name
                if (categoryName not in categoriesList):
                    categoriesList.append(categoryName)

                complexityIndexDTO = ComplexityIndexDTO(repr(key), float(value), 
                                                        type="sentence",
                                                        paragraph_index=paragraph_id,
                                                        sentence_index=sentence_id)
                # complexityIndex[categoryName] = complexityIndexDTO
                if categoryName not in complexityIndices:
                    complexityIndices[categoryName] = []
                complexityIndices[categoryName].append(complexityIndexDTO)

    # iterate through complexity index array
    complexityIndicesResponse = [
        ComplexityIndicesDTO(category, indices) 
        for category, indices in complexityIndices.items()
    ]
    texts = [
        [sentence.text for sentence in paragraph.components]
        for paragraph in document.components
    ]
    
    textualComplexityDataDTO = TextualComplexityDataDTO(
        languageString, texts, categoriesList, complexityIndicesResponse)

    textualComplexityResponse = TextualComplexityResponse(
        textualComplexityDataDTO, "", True)
    jsonString = textualComplexityResponse.toJSON()
    # print(textualComplexityResponse)
    # jsonString = json.dumps(textualComplexityResponse, default=TextualComplexityResponse.dumper, indent=2)

    return jsonString
