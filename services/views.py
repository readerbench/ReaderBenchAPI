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
from services.models import Job, Language


@api_view(['POST'])
@permission_classes([AllowAny])
def process_cscl(request):
    from services import cscl
    lang = request.data["lang"]
    result = cscl.process_conv(request.FILES.get("file"), lang)
    
    return JsonResponse(result, safe=False)

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
    try:
        user_id = request.user.id if request.user.id is not None else 1
        jobs = [
            job.to_dict()
            for job in Job.objects.filter(user_id=user_id).order_by("-id").all()
        ]
        return JsonResponse({"jobs": jobs}, safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving jobs'}, status=500)

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
    
@api_view(['POST'])
@permission_classes([AllowAny])
def add_sentiment_job(request):
    try:
        text = request.data["text"]
        job = Job()
        job.type_id = JobTypeEnum.SENTIMENT.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error adding sentiment job'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def add_keywords_job(request):
    try:
        text = request.data["text"]
        lang = int(request.data["lang"])
        job = Job()
        job.type_id = JobTypeEnum.KEYWORDS.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text, "lang": lang})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error adding keywords job'}, status=500)
    
@api_view(['POST'])
@permission_classes([AllowAny])
def add_offensive_job(request):
    try:
        text = request.data["text"]
        job = Job()
        job.type_id = JobTypeEnum.OFFENSIVE.value
        job.status_id = JobStatusEnum.PENDING.value
        job.user_id = request.user.id if request.user.id is not None else 1
        job.params = json.dumps({"text": text})
        job.save()
        return JsonResponse({"id": job.id})
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error adding offensive job'}, status=500)