import csv
import json
import os
from shutil import rmtree

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from pipeline.models import Model
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
        job.dataset = dataset
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
        models = Model.objects.filter(dataset__user_id=user_id).order_by("id").all()
        result = [
            {
                "id": model.id,
                "dataset": model.dataset.name,
                "job_id": model.job_id,
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

@api_view(['POST'])
@permission_classes([AllowAny])
def delete_model(request, model_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        model = get_object_or_404(Model, id=model_id)
        if model.job.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Unauthorized access'}, status=403)
        if os.path.exists(f"data/models/{model_id}"):
            rmtree(f"data/models/{model_id}")
        model.delete()
        return JsonResponse({"success": True})
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'delete_model_operation_failed', 'message': 'Error deleting model'}, status=500)
 

@api_view(['POST'])
@permission_classes([AllowAny])
def model_predict(request, model_id):
    # try:
    user_id = request.user.id if request.user.id is not None else 1
    model = get_object_or_404(Model, id=model_id)
    if model.job.user_id != user_id:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Model not owned be user'}, status=403)
    csvfile = request.FILES.get("csvfile")
    if csvfile is None:
        from rb.core.lang import Lang
        from pipeline.enums import ModelTypeEnum
        from pipeline.parallel import build_features
        
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

@api_view(['POST'])
@permission_classes([AllowAny])
def model_feature_importances(request, model_id):
    try:
        user_id = request.user.id if request.user.id is not None else 1
        model = get_object_or_404(Model, id=model_id)
        if model.job.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Model not owned be user'}, status=403)
        from rb.core.lang import Lang
        from pipeline.enums import ModelTypeEnum
        if model.type_id != ModelTypeEnum.XGBOOST.value:
            return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Feature importance can only be computed on XGBoost models'}, status=403)
        import xlsxwriter
        import io
        buffer = io.BytesIO()
        with xlsxwriter.Workbook(buffer) as workbook:
            worksheet = workbook.add_worksheet()       
            merge_format = workbook.add_format({'align': 'center'})
            dataset = model.dataset
            tasks = []
            for i in range(dataset.num_cols):
                with open(f"data/datasets/{dataset.id}/task_{i}.json", "rt") as f:
                    obj = json.load(f)
                    task = Task(obj=obj)
                    tasks.append(task)
            lang = Lang[dataset.lang.label]
            multipredictor = ModelTypeEnum(model.type_id).predictor()(lang, tasks)
            multipredictor.load(model)
            for i, predictor in enumerate(multipredictor.predictors):
                worksheet.merge_range(0, i * 2, 0, i * 2 + 1, predictor.task.name, merge_format)
                worksheet.write_string(1, i * 2, "Feature")
                worksheet.write_string(1, i * 2 + 1, "Importance")
                importances = sorted(zip(predictor.task.features, predictor.model.feature_importances_), key=lambda x: x[1], reverse=True)
                for j, entry in enumerate(importances):
                    worksheet.write_string(j+2, i * 2, entry[0])
                    worksheet.write_number(j+2, i * 2 + 1, entry[1])
        buffer.seek(0)
        data = buffer.getvalue()
        response = HttpResponse(data, content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        response['Content-Disposition'] = f'attachment; filename="Feature importance.xlsx"'
        response['Access-Control-Expose-Headers'] = '*'
        return response
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_result_operation_failed', 'message': 'Error returning results'}, status=500)
