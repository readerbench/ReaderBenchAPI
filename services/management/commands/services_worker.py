import json
import time
from datetime import datetime

import torch
import tensorflow as tf
from django.core.management.base import BaseCommand
from rb.processings.diacritics.DiacriticsRestoration import \
    DiacriticsRestoration

from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Job
from services.qgen.answer_generation import generate_answers
from services.qgen.distractors_gen import generate_distractors


class Command(BaseCommand):
    help = 'Runs services jobs'

    def handle(self, *args, **options):
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[0], 'GPU')
        job_types = [
            JobTypeEnum.CSCL.value, JobTypeEnum.OFFENSIVE.value, JobTypeEnum.SENTIMENT.value, JobTypeEnum.DIACRITICS.value,
            JobTypeEnum.INDICES.value, JobTypeEnum.ANSWER_GEN.value, JobTypeEnum.TEST_GEN.value,
        ]
        while True:
            job = Job.objects \
                .filter(status_id=JobStatusEnum.PENDING.value) \
                .filter(type_id__in=job_types) \
                .exclude(params="{}") \
                .first()
            if job is None:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    raise
                continue
            print(f"Starting {JobTypeEnum(job.type_id).name} job {job.id}...")
            params = json.loads(job.params)
            t1 = datetime.now()
            try:
                if job.type_id == JobTypeEnum.ANSWER_GEN.value:
                    text = params["text"]
                    result = generate_answers(text)
                    job.results = json.dumps(result)
                    job.status_id = JobStatusEnum.FINISHED.value
                    t2 = datetime.now()
                    job.elapsed_seconds = (t2 - t1).seconds
                    job.save()
                elif job.type_id == JobTypeEnum.TEST_GEN.value:
                    text = params["text"]
                    answers = params["answers"]
                    result = generate_distractors(text, answers)
                    job.results = json.dumps(result)
                    job.status_id = JobStatusEnum.FINISHED.value
                    t2 = datetime.now()
                    job.elapsed_seconds = (t2 - t1).seconds
                    job.save()
                elif job.type_id == JobTypeEnum.DIACRITICS.value:
                    text = params["text"]
                    model = DiacriticsRestoration()
                    result = model.process_string(text)
                    job.results = json.dumps(result)
                    job.status_id = JobStatusEnum.FINISHED.value
                    t2 = datetime.now()
                    job.elapsed_seconds = (t2 - t1).seconds
                    job.save()
                    
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                t2 = datetime.now()
                print(ex)
                job.status_id = JobStatusEnum.ERROR.value
                job.elapsed_seconds = (t2 - t1).seconds
                job.save()

            print(f"{JobTypeEnum(job.type_id).name} job {job.id} finished")
            

