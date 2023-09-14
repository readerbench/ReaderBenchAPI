import csv
import io
import json
import os
from shutil import rmtree
import xlsxwriter
import zipfile

from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, render

from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

from pipeline.models import Dataset, Model
from pipeline.task import TargetType, Task
from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Job, Language


        
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
        pipeline_job = Job.objects.filter(type_id=JobTypeEnum.PIPELINE.value, dataset_id=dataset_id).order_by("-id").first()
        processed = 0 if pipeline_job is None else pipeline_job.status_id
        indices_job = Job.objects.filter(type_id=JobTypeEnum.INDICES.value, dataset_id=dataset_id).order_by("-id").first()
        indices = 0 if indices_job is None else indices_job.status_id
        result = {
            "id": dataset.id,
            "name": dataset.name,
            "language": dataset.lang_id,
            "number_of_tasks": dataset.num_cols,
            "number_of_entries": dataset.num_rows,
            "processed": processed,
            "pipeline_job_id": pipeline_job.id if pipeline_job is not None else None,
            "indices_job_id": indices_job.id if indices_job is not None else None,
            "indices": indices,
        }
        return JsonResponse(result, safe=False)
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error while retrieving datasets'}, status=500)

@api_view(['POST'])
@permission_classes([AllowAny])
def delete_dataset(request, dataset_id):
    try:
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
    except Exception as ex:
        return JsonResponse({'status': 'ERROR', 'error_code': 'get_operation_failed', 'message': 'Error deleting dataset'}, status=500)


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
def add_indices_job(request, dataset_id):
    try:
        dataset = get_object_or_404(Dataset, id=dataset_id)
        user_id = request.user.id if request.user.id is not None else 1
        if dataset.user_id != user_id:
            return JsonResponse({'status': 'ERROR', 'error_code': 'add_job_operation_failed', 'message': 'Forbidden'}, status=403)
        job = Job()
        job.user_id = user_id
        job.type_id = JobTypeEnum.INDICES.value
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
    try:
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
    except Exception as ex:
        print(ex)
        return JsonResponse({'status': 'ERROR', 'error_code': 'predict_operation_failed', 'message': 'Prediction error'}, status=500)
 

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
        elif job.type_id == JobTypeEnum.INDICES.value:
            response = HttpResponse(
                content_type="application/x-zip-compressed",
                headers={"Content-Disposition": 'attachment; filename="indices.zip"'},
            )
            docs = {}
            for filename in os.listdir(f"data/datasets/{job.dataset_id}/indices"):
                if not filename.endswith(".json"):
                    continue
                with open(f"data/datasets/{job.dataset_id}/indices/{filename}", "rt") as f:
                    docs[int(filename.split(".json")[0])] = json.load(f)
            docs = [docs[key] for key in sorted(docs.keys())]
            with zipfile.ZipFile(response, "w") as zip:
                buffer = io.BytesIO()
                with xlsxwriter.Workbook(buffer) as workbook:
                    worksheet = workbook.add_worksheet()       
                    worksheet.write_string(0, 0, "Index")
                    worksheet.write_string(0, 1, "Text")
                    features = list(sorted(key for key in docs[0]["indices"][0].keys()))
                    for i, feature in enumerate(features):
                        worksheet.write_string(0, i+2, feature)
                    for i, doc in enumerate(docs):
                        worksheet.write_number(i+1, 0, i)
                        worksheet.write_string(i+1, 1, doc["elements"][0]["text"])
                        for j, feature in enumerate(features):
                            if doc["indices"][0][feature] is not None:
                                worksheet.write_number(i+1, j+2, doc["indices"][0][feature])
                buffer.seek(0)
                data = buffer.getvalue()
                zip.writestr("doc_indices.xlsx", data)
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
