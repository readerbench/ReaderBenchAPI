import json
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated

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
 
    

