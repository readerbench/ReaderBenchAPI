from django.db import models

from services.models import Dataset, Job

# Create your models here.
class ModelType(models.Model):
    label = models.CharField(max_length=20)

class Model(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    type = models.ForeignKey(ModelType, on_delete=models.CASCADE)
    params = models.TextField(default="{}")
    metrics = models.TextField(default="{}")
