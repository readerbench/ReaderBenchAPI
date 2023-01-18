import csv
import datetime
from os import makedirs, rmdir
import os
from typing import Dict
import zipfile

import rb
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cscl_parser import load_from_xml
from rb.core.text_element import TextElement, TextElementType
from rb.processings.cscl.participant_evaluation import (
    evaluate_interaction, evaluate_involvement, evaluate_textual_complexity,
    perform_sna)
from rb.processings.diacritics.DiacriticsRestoration import \
    DiacriticsRestoration
from rb.processings.keywords.keywords_extractor import extract_keywords
from rb.similarity.similar_concepts import get_similar_concepts
from rb.similarity.transformers_encoder import TransformersEncoder
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.utils import str_to_lang
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from services.feedback import feedback
from services.models import Dataset, Language
from services.readme_misc.fluctuations import calculate_indices
from services.readme_misc.keywords import *
from services.readme_misc.similarity import get_hypernymes_grouped_by_synset
from services.readme_misc.utils import find_mistakes_intervals
from services.russian_a_vs_b.ru_a_vs_b import RussianAvsB
from services.subject_predicate.correct import (language_correct,
                                                ro_language_correct)
from services.syllables.syllables import syllabify


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

@api_view(['POST'])
@permission_classes([AllowAny])
def process_cscl(request):
    conv_dict = load_from_xml(request.FILES.get("file"))
    lang = request.data["lang"]
    lang = str_to_lang(lang)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, None)
    conv = Conversation(lang, conv_dict, apply_heuristics=False)
    model.encode(conv)
    conv.graph = CnaGraph(conv, models=[model], pairwise=False, window=30)
    evaluate_interaction(conv)
    evaluate_involvement(conv)
    participant_graph = perform_sna(conv, False)

    contr_kb = {
        contribution: 0
        for contribution in conv.components
    }
    contr_in = {
        contribution: 0
        for contribution in conv.components
    }
    contr_out = {
        contribution: 0
        for contribution in conv.components
    }
    contr_edges = []
    for u, v, w in conv.graph.filtered_graph.edges.data('weight', default=None):
        if isinstance(u, Contribution) and isinstance(v, Contribution) and u.index < v.index:
            contr_in[u] += w
            contr_out[v] += w
            contr_edges.append({
                "source": u.index, 
                "target": v.index,
                "weight": w
            })
            if u.get_participant() != v.get_participant():
                contr_kb[u] += w
                contr_kb[v] += w

    result = {
        "graph": {
            "participants": [p.get_id() for p in participant_graph.nodes],
            "edges": [
                {
                    "source": a.get_id(), 
                    "target": b.get_id(),
                    "weight": w
                }
                for a, b, w in participant_graph.edges.data("weight")
            ]
        }, 
        "participants": {
            p.get_id(): {str(index): value for index, value in p.indices.items()}
            for p in conv.get_participants()
        },
        "contributions": [
            {
                "id": c.index,
                "text": c.text,
                "participant": c.get_participant().get_id(),
                "ref": c.parent_contribution.index if c.parent_contribution is not None else None,
                "time": c.timestamp.isoformat(),
                "importance": conv.graph.importance[c],
                "social_kb": contr_kb[c],
                "in_degree": contr_in[c],
                "out_degree": contr_out[c]
            }
            for c in conv.components
        ],
        "contribution_edges": contr_edges
    }
    
    return JsonResponse(result, safe=False)

@api_view(['POST'])
@permission_classes([AllowAny])
def add_dataset(request):
    try:
        data = request.FILES.get("zipfile")
        targets = request.FILES.get("csvfile")
        lang_id = request.data["lang"]
        lang = get_object_or_404(Language, pk=lang_id)
        name = request.POST["name"]
        task = request.POST ["task"]
        dataset = Dataset()
        dataset.name = name
        dataset.task = task
        dataset.user_id = 1
        dataset.lang = lang
        dataset.save()
        os.makedirs(f"data/datasets/{dataset.pk}/texts")
        with open(f"data/datasets/{dataset.pk}/targets.csv", "wb") as f:
            for chunk in targets.chunks():
                f.write(chunk)
        with open(f"data/datasets/{dataset.pk}/targets.csv", "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            dataset.num_cols = len(header) - 1
        with zipfile.ZipFile(data) as f:
            count = 0
            for zip_info in f.infolist():
                if zip_info.filename[-1] == '/':
                    continue
                zip_info.filename = os.path.basename(zip_info.filename)
                f.extract(zip_info, f"data/datasets/{dataset.pk}/texts")
                count += 1
            dataset.num_rows = count
        dataset.save()
        return JsonResponse({"id": dataset.pk})
    except Exception as ex:
        if 'dataset' in locals() and dataset.pk is not None:
            rmdir(f"data/datasets/{dataset.pk}")
            dataset.delete()
        return JsonResponse({'status': 'ERROR', 'error_code': 'add_operation_failed', 'message': 'The dataset could not be saved'}, status=500)
 
@api_view(['POST'])
@permission_classes([AllowAny])
def get_datasets(request):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        datasets = Dataset.objects.filter(user_id=user_id).all()
        datasets = [
            {
                "id": dataset.id,
                "name": dataset.name,
                "language": dataset.lang_id,
                "number_of_tasks": dataset.num_cols,
                "number_of_entries": dataset.num_rows,
            }
            for dataset in datasets
        ]
        return JsonResponse({"datasets": datasets}, safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving datasets'}, status=500)
 
@api_view(['POST'])
@permission_classes([AllowAny])
def get_languages(request):
    try:
        languages = [
            {
                "id": lang.id,
                "label": lang.label,
            }
            for lang in Language.objects.all()
        ]
        return JsonResponse({"languages": languages}, safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving datasets'}, status=500)
    
