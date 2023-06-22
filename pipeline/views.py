import csv
import json
import os

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render
from joblib import Parallel, delayed
import numpy as np
from rb import Lang
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from pipeline.enums import ModelTypeEnum
from pipeline.models import Model
from pipeline.parallel import build_features
from pipeline.task import TargetType, Task
from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Dataset, Job


# Create your views here.
@api_view(['POST'])
@permission_classes([AllowAny])
def process_dataset(request, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        user_id = request.user.id if request.user.id is not None else 1
        if dataset.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'add_job_operation_failed', 'message': 'Forbidden'}, status=403)
        job = Job()
        job.user_id = user_id
        job.type_id = JobTypeEnum.PIPELINE.value
        job.status_id = JobStatusEnum.PENDING.value
        job.params = json.dumps({"dataset_id": dataset_id})
        job.save()
        result = {"id": job.id}
        return JsonResponse(result, safe=False)
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'add_job_operation_failed', 'message': 'Error processing dataset'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def get_models(request):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        models = Model.objects.filter(dataset__user_id=user_id).all()
        result = [
            {
                "id": model.id,
                "dataset": model.dataset.name,
                "metrics": model.metrics,
                "params": model.params,
                "type": model.type.label
            }
            for model in models
        ]
        return JsonResponse(result, safe=False)
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_models_operation_failed', 'message': 'Error returning models'}, status=500)
 

# Create your views here.
@api_view(['POST'])
@permission_classes([AllowAny])
def model_predict(request, model_id):
    # try:
    model = get_object_or_404(Model, id=model_id)
    csvfile = request.FILES.get("csvfile")
    if csvfile is None:
        dataset = model.dataset
        tasks = []
        for i in range(dataset.num_cols):
            with open(f"data/datasets/{dataset.id}/task_{i}.json", "rt") as f:
                obj = json.load(f)
                task = Task(obj=obj)
                tasks.append(task)
        lang = Lang[dataset.lang.label]
        predictor = ModelTypeEnum(model.type_id).predictor()(lang, tasks)
        text = request.data["text"]
        all_features = [build_features(text, lang)]
        texts = [text]
        predictions = predictor.predict(model, texts, all_features)
        result = []
        for task, pred in zip(tasks, predictions):
            if task.type is TargetType.STR:
                output = json.dumps({c: round(float(p), 2) for c, p in zip(task.classes, pred[0])})
            else:
                output = round(float(pred), 2)
            result.append({
                "task": task.name,
                "result": output,
            })
        return JsonResponse(result, safe=False)
    else:
        job = Job()
        job.user_id = request.user.id if request.user.id is not None else 1
        job.type_id = JobTypeEnum.PREDICT.value
        job.params = "{}"
        job.status_id = JobStatusEnum.PENDING.value
        job.save()
        os.makedirs(f"data/jobs/{job.id}")
        with open(f"data/jobs/{job.id}/input.csv", "wb") as f:
            for chunk in csvfile.chunks():
                f.write(chunk)
        job.params = json.dumps({"model_id": model_id})
        job.save()
        result = {"id": job.id}
        return JsonResponse(result, safe=False)
    # except Exception as ex:
    #     print(ex)
    #     return JsonResponse({'status': 'ERROR', 'error_code': 'predict_operation_failed', 'message': 'Prediction error'}, status=500)
 

@api_view(['POST'])
@permission_classes([AllowAny])
def get_result(request, job_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        job = get_object_or_404(Job, id=job_id)
        if job.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Job not owned be user'}, status=403)
        if job.status_id != JobStatusEnum.FINISHED.value:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Job not finished'}, status=404)
        if job.type_id == JobTypeEnum.PREDICT.value:
            response = HttpResponse(
                content_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="Results.csv"'},
            )
            with open(f"data/jobs/{job.id}/output.csv", "rt") as f:
                reader = csv.reader(f)
                writer = csv.writer(response)
                for row in reader:
                    writer.writerow(row)
            return response
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Error returning results'}, status=500)
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Error returning results'}, status=500)

