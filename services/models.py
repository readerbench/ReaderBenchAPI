import json
from django.db import models
from django.conf import settings

# Create your models here.
class Language(models.Model):
    label = models.CharField(max_length=20)
    
class JobType(models.Model):
    label = models.CharField(max_length=20)

class JobStatus(models.Model):
    label = models.CharField(max_length=20)

class Job(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    type = models.ForeignKey(JobType, on_delete=models.CASCADE)
    status = models.ForeignKey(JobStatus, on_delete=models.CASCADE)
    dataset = models.ForeignKey("pipeline.Dataset", on_delete=models.CASCADE, null=True)
    submit_time = models.DateTimeField(auto_now_add=True)
    elapsed_seconds = models.IntegerField(default=0)
    params = models.TextField(default="{}")
    results = models.TextField(default="{}")

    def to_dict(self):
        from pipeline.models import Model
        params = json.loads(self.params)
        dataset = None
        if self.dataset_id is not None:
            dataset = {
                "id": self.dataset_id,
                "name": self.dataset.name,
            }
        if "model_id" in params:
            try:
                model = Model.objects.get(id=params["model_id"])
                params["model"] = model.type.label
            except:
                params["model"] = None
            del params["model_id"]
        return {
            "id": self.id,
            "status": self.status_id,
            "type": self.type_id,
            "dataset": dataset,
            "params": params,
            "results": json.loads(self.results),
            "submit_time": self.submit_time.timestamp(),
            "elapsed_time": self.elapsed_seconds,
        }
            