import csv
import json

from django.http import JsonResponse
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
 
# Create your views here.
@api_view(['POST'])
@permission_classes([AllowAny])
def model_predict(request, model_id):
    # try:
    model = get_object_or_404(Model, id=model_id)
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
    texts = [text]
    with Parallel(n_jobs=8, prefer="processes", verbose=100) as parallel:
        all_features = parallel( \
            delayed(build_features)(text, lang) \
            for text in texts)
    predictions = predictor.predict(model, text, all_features)
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
    # except Exception as ex:
    #     print(ex)
    #     return JsonResponse({'status': 'ERROR', 'error_code': 'add_job_operation_failed', 'message': 'Error processing dataset'}, status=500)
 


