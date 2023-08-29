from django.conf import settings
from django.db import models

from services.models import Job, Language


class Dataset(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    task = models.CharField(max_length=50)
    lang = models.ForeignKey(Language, on_delete=models.CASCADE)
    num_rows = models.IntegerField(default=0)
    num_cols = models.IntegerField(default=0)

class ModelType(models.Model):
    label = models.CharField(max_length=20)

class Model(models.Model):
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    job = models.ForeignKey(Job, on_delete=models.CASCADE)
    type = models.ForeignKey(ModelType, on_delete=models.CASCADE)
    params = models.TextField(default="{}")
    metrics = models.TextField(default="{}")
