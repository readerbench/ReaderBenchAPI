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