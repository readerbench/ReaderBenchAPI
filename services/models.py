import json
from django.db import models
from django.conf import settings

# Create your models here.
class Language(models.Model):
    label = models.CharField(max_length=20)
    
class Dataset(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    task = models.CharField(max_length=50)
    lang = models.ForeignKey(Language, on_delete=models.CASCADE)
    num_rows = models.IntegerField(default=0)
    num_cols = models.IntegerField(default=0)

class JobType(models.Model):
    label = models.CharField(max_length=20)

class JobStatus(models.Model):
    label = models.CharField(max_length=20)

class Job(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    type = models.ForeignKey(JobType, on_delete=models.CASCADE)
    status = models.ForeignKey(JobStatus, on_delete=models.CASCADE)
    submit_time = models.DateTimeField(auto_now_add=True)
    elapsed_seconds = models.IntegerField(default=0)
    params = models.TextField(default="{}")
    results = models.TextField(default="{}")

    def to_dict(self):
        from pipeline.models import Model
        params = json.loads(self.params)
        if "dataset_id" in params:
            params["dataset"] = Dataset.objects.get(id=params["dataset_id"]).name
            del params["dataset_id"]
        if "model_id" in params:
            params["model"] = Model.objects.get(id=params["model_id"]).type.label
            del params["model_id"]
        return {
            "id": self.id,
            "status": self.status_id,
            "type": self.type_id,
            "params": params,
            "results": json.loads(self.results),
            "submit_time": self.submit_time.timestamp(),
            "elapsed_time": self.elapsed_seconds,
        }
            