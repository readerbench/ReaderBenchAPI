import json
import os
import time

from django.core.management.base import BaseCommand
from rb import Lang

from services.models import Job, Language
from services.enums import JobStatusEnum, JobTypeEnum
from services.pipeline import process

class Command(BaseCommand):
    help = 'Runs pipeline jobs'

    def handle(self, *args, **options):
        while True:
            job = Job.objects.filter(status_id=JobStatusEnum.PENDING.value, type_id=JobTypeEnum.PIPELINE.value).first()
            if job is None:
                time.sleep(1)
                continue
            print(f"Starting pipeline job {job.id}...")
            process(job)
            print(f"Pipeline job {job.id} finished")
            

