from typing import Dict

from django.http import JsonResponse
from django.shortcuts import render

from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.text_element import TextElement, TextElementType
from rb.similarity.transformers_encoder import TransformersEncoder
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.similar_concepts import get_similar_concepts
from rb.utils.utils import str_to_lang
from rb.processings.keywords.keywords_extractor import extract_keywords
from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration
import rb

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from services.subject_predicate.correct import ro_language_correct, language_correct
from services.feedback import feedback
from services.readme_misc.similarity import get_hypernymes_grouped_by_synset
from services.readme_misc.universal_text_extractor import extract_raw_text
from services.readme_misc.fluctuations import calculate_indices
from services.readme_misc.keywords import *
from services.readme_misc.utils import find_mistakes_intervals
from services.syllables.syllables import syllabify
from services.russian_a_vs_b.ru_a_vs_b import RussianAvsB


def build_text_element_result(elem: TextElement) -> Dict:
    if elem.depth < TextElementType.SENT.value:
        return {
            "text": elem.text
        }
    return {
        "text": elem.text,
        "indices": sorted([
                (str(index), value) 
                for index, value in elem.indices.items()
            ], 
            key=lambda x: x[0]),
        "children": [
            build_text_element_result(child) 
            for child in elem.components
        ]
    }


# Create your views here.
@api_view(['POST'])
@permission_classes([AllowAny])
def get_indices(request):
    text = request.data["text"]
    lang = request.data["lang"]
    lang = str_to_lang(lang)
    doc = Document(lang=lang, text=text)
    encoder: TransformersEncoder = create_vector_model(lang, VectorModelType.TRANSFORMER, None)
    encoder.encode(doc)
    graph = CnaGraph(doc, models=[encoder])
    compute_indices(doc, graph)
    response = build_text_element_result(doc)
    return JsonResponse(response, safe=False)


# aici definesc functii care vor fi accesibile prin url-uri
@api_view(['POST'])
@permission_classes([AllowAny])
def ro_correct_text(request):
    data = request.data
    text = data['text']
    ro_model = rb.parser.spacy_parser.SpacyParser.get_instance().get_model(Lang.RO)
    try:
        errors = ro_language_correct(text, ro_model)
        return JsonResponse({'mistakes': errors})
    except: # TODO: handle exception
        return JsonResponse({'mistakes': []})


@api_view(['POST'])
@permission_classes([AllowAny])
def en_correct_text(request):
    text = request.data.get('text', '')
    return JsonResponse(language_correct(text, Lang.EN, 'TODO_hunspell/en_US.dic', 'TODO_hunspell/en_US.aff'))

@api_view(['POST'])
@permission_classes([AllowAny])
def feedbackPost(request):
    data = request.data
    text = data['text']
    lang = request.data["lang"]
    lang = str_to_lang(lang)
    if lang != Lang.RO:
        return JsonResponse({'error': 'Language not supported; Only ro supported'})
    response = dict()
    doc_indices = feedback.compute_textual_indices(text, lang)
    response['feedback'] = feedback.automatic_feedback(doc_indices)
    response['score'] = feedback.automatic_scoring(doc_indices)
    return JsonResponse(response)


@api_view(['POST'])
@permission_classes([AllowAny])
def fluctuations(request):
    data = request.data
    text = data['text']
    lang = data.get('lang', 'ro')
    if lang == 'ro':
        response = calculate_indices(text, lang=Lang.RO)
    else:
        response = calculate_indices(text, lang=Lang.EN)

    return JsonResponse(response, safe=False)


@api_view(['POST'])
@permission_classes([AllowAny])
def keywords(request):
    data = request.data
    text = data['text']
    lang = str_to_lang(data.get('lang', 'en'))
    extracted_keywords = extract_keywords(text=text, lang=lang)
    return JsonResponse(transform_for_visualization(extracted_keywords, lang))


@api_view(['POST'])
@permission_classes([AllowAny])
def keywordsHeatmap(request):
    data = request.data
    text = data['text']
    granularity = data.get('granularity', TextElementType.SENT)
    if granularity == "sentence":
        granularity = TextElementType.SENT
    else:
        granularity = TextElementType.BLOCK
    lang = str_to_lang(data['lang'])

    return JsonResponse(keywords_heatmap(text=text, lang=lang, granularity=granularity))


@api_view(['POST'])
@permission_classes([AllowAny])
def similar_concepts(request):
    data = request.data
    word = data['text']
    lang = str_to_lang(data['lang'])
    similar_concept = get_similar_concepts(word, lang)
    return JsonResponse(similar_concept, safe=False)


@api_view(['POST'])
@permission_classes([AllowAny])
def get_hypernyms(request):
    data = request.data
    word = data['text']
    lang = str_to_lang(data['lang'])
    hypernyms = get_hypernymes_grouped_by_synset(word, lang)
    return JsonResponse(hypernyms)


@api_view(['POST'])
@permission_classes([AllowAny])
def syllables(request):
    data = request.data
    text = data['text']
    ro_model = rb.parser.spacy_parser.SpacyParser.get_instance().get_model(Lang.RO)
    try:
        return JsonResponse(syllabify(text, ro_model), safe=False)
    except: # TODO handle exception
        return JsonResponse({})
    

@api_view(['POST'])
@permission_classes([AllowAny])
def restore_diacritics(request):
    data = request.data
    text = data['text']
    dr = DiacriticsRestoration()
    restored_text = dr.process_string(text, mode="replace_missing")
    mistake_intervals = find_mistakes_intervals(text, restored_text)
    return JsonResponse({'restored': restored_text, 'mistakes': mistake_intervals})


@api_view(['POST'])
@permission_classes([AllowAny])
def clasify_aes(request):
    data = request.data
    text = data['text']
    r = RussianAvsB(model_path="services/binaries/ru_aes/AvsB_ru", rb_indices_path="services/binaries/ru_aes/ru_indices.list")
    return JsonResponse({'class': r.predict(text)})