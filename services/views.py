import csv
import json
import os
from shutil import rmtree
import zipfile

from typing import Dict

from django.http import JsonResponse
from django.shortcuts import get_object_or_404

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from pipeline.models import Model
from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Dataset, Job, Language

# def build_text_element_result(elem: TextElement) -> Dict:
#     if elem.depth < TextElementType.SENT.value:
#         return {
#             "text": elem.text
#         }
#     return {
#         "text": elem.text,
#         "indices": sorted([
#                 (str(index), value) 
#                 for index, value in elem.indices.items()
#             ], 
#             key=lambda x: x[0]),
#         "children": [
#             build_text_element_result(child) 
#             for child in elem.components
#         ]
#     }


# Create your views here.
# @api_view(['POST'])
# @permission_classes([AllowAny])
# def get_indices(request):
#     text = request.data["text"]
#     lang = request.data["lang"]
#     lang = str_to_lang(lang)
#     doc = Document(lang=lang, text=text)
#     encoder: TransformersEncoder = create_vector_model(lang, VectorModelType.TRANSFORMER, None)
#     encoder.encode(doc)
#     graph = CnaGraph(doc, models=[encoder])
#     compute_indices(doc, graph)
#     response = build_text_element_result(doc)
#     return JsonResponse(response, safe=False)


# # aici definesc functii care vor fi accesibile prin url-uri
# @api_view(['POST'])
# @permission_classes([AllowAny])
# def ro_correct_text(request):
#     data = request.data
#     text = data['text']
#     ro_model = rb.parser.spacy_parser.SpacyParser.get_instance().get_model(Lang.RO)
#     try:
#         errors = ro_language_correct(text, ro_model)
#         return JsonResponse({'mistakes': errors})
#     except: # TODO: handle exception
#         return JsonResponse({'mistakes': []})


# @api_view(['POST'])
# @permission_classes([AllowAny])
# def en_correct_text(request):
#     text = request.data.get('text', '')
#     return JsonResponse(language_correct(text, Lang.EN, 'TODO_hunspell/en_US.dic', 'TODO_hunspell/en_US.aff'))

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
def clasify_aes(request):
    data = request.data
    text = data['text']
    r = RussianAvsB(model_path="services/binaries/ru_aes/AvsB_ru", rb_indices_path="services/binaries/ru_aes/ru_indices.list")
    return JsonResponse({'class': r.predict(text)})

@api_view(['POST'])
@permission_classes([AllowAny])
def process_cscl(request):
    lang = request.data["lang"]
    result = cscl.process_conv(request.FILES.get("file"), lang)
    
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
        task = request.POST["task"]
        dataset = Dataset()
        dataset.name = name
        dataset.task = task
        dataset.user_id = 1
        dataset.lang = lang
        dataset.save()
        os.makedirs(f"data/datasets/{dataset.pk}")
        with open(f"data/datasets/{dataset.pk}/targets.csv", "wb") as f:
            for chunk in targets.chunks():
                f.write(chunk)
        with open(f"data/datasets/{dataset.pk}/targets.csv", "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            dataset.num_cols = len(header) - 1
            dataset.num_rows = sum(1 for row in reader)
            dataset.save()
        if data is not None:
            os.makedirs(f"data/datasets/{dataset.pk}/texts")
            with zipfile.ZipFile(data) as f:
                for zip_info in f.infolist():
                    if zip_info.filename[-1] == '/':
                        continue
                    zip_info.filename = os.path.basename(zip_info.filename)
                    f.extract(zip_info, f"data/datasets/{dataset.pk}/texts")
        return JsonResponse({"id": dataset.pk})
    except Exception as ex:
        print(ex)
        if 'dataset' in locals() and dataset.pk is not None:
            rmtree(f"data/datasets/{dataset.pk}")
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
def get_dataset(request, dataset_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        dataset = get_object_or_404(Dataset, id=dataset_id)
        if dataset.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Unauthorized access'}, status=403)
        job = Job.objects.filter(type_id=JobTypeEnum.PIPELINE.value, dataset_id=dataset_id).order_by("-id").first()
        processed = 0 if job is None else job.status_id
        result = {
            "id": dataset.id,
            "name": dataset.name,
            "language": dataset.lang_id,
            "number_of_tasks": dataset.num_cols,
            "number_of_entries": dataset.num_rows,
            "processed": processed,
            "indices": False,
        }
        return JsonResponse(result, safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving datasets'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def delete_dataset(request, dataset_id):
    # try:
    user_id = request.user.id if request.user.id is not None else 1
    dataset = get_object_or_404(Dataset, id=dataset_id)
    if dataset.user_id != user_id:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Unauthorized access'}, status=403)
    for model in Model.objects.filter(dataset_id=dataset_id).all():
        rmtree(f"data/models/{model.id}")
        model.delete()
    rmtree(f"data/datasets/{dataset_id}")
    dataset.delete()    
    return JsonResponse({"success": True}, safe=False)
    # except Exception as ex:
    #     return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error deleting dataset'}, status=500)


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
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving languages'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def get_jobs(request):
    # try:
    user_id = request.user.id if request.user.id is not None else 1
    jobs = [
        job.to_dict()
        for job in Job.objects.filter(user_id=user_id).all()
    ]
    return JsonResponse({"jobs": jobs}, safe=False)
    # except Exception as ex:
    #     return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving jobs'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def get_job(request, job_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        job = get_object_or_404(Job, id=job_id)
        if job.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Unauthorized access'}, status=403)
        return JsonResponse(job.to_dict(), safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving job'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def delete_job(request, job_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        job = get_object_or_404(Job, id=job_id)
        if job.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Unauthorized access'}, status=403)
        for model in Model.objects.filter(job_id=job_id).all():
            if os.path.exists(f"data/models/{model.id}"):
                rmtree(f"data/models/{model.id}")
            model.delete()
        if os.path.exists(f"data/jobs/{job_id}"):
            rmtree(f"data/jobs/{job_id}")
        job.delete()
        return JsonResponse({"success": True})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'delete_job_operation_failed', 'message': 'Error while deleting job'}, status=500)


@api_view(['POST'])
@permission_classes([AllowAny])
def get_potential_answers(request):
    try:
        text = request.data["text"]
        job = Job()
        job.type_id = JobTypeEnum.ANSWER_GEN.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error generating answers'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def generate_test(request):
    try:
        text = request.data["text"]
        answers = request.data["answers"]
        job = Job()
        job.type_id = JobTypeEnum.TEST_GEN.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text, "answers": answers})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error generating test'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def restore_diacritics(request):
    try:
        text = request.data["text"]
        job = Job()
        job.type_id = JobTypeEnum.DIACRITICS.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error restoring diacritics'}, status=500)