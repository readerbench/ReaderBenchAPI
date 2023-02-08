import csv
import json
import os
import time

from django.core.management.base import BaseCommand
from joblib import Parallel, delayed
from rb import Lang

from pipeline.parallel import build_features
from pipeline.preprocessing import filter_rare, generator, split
from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Dataset, Job, Language


def process(job: Job):
    job.status_id = JobStatusEnum.IN_PROGRESS.value
    job.save()
    try:
        params = json.loads(job.params)
        dataset = Dataset.objects.get(id=params["dataset_id"])
        root = f"data/datasets/{dataset.id}"
        lang = Lang[dataset.lang.label]
        # split(dataset)
        # for partition in ["train", "val", "test"]:
        #     features = Parallel(n_jobs=8, prefer="processes")( \
        #         delayed(build_features)(row[0], lang) \
        #         for row in generator(dataset, partition))
        #     with open(f"{root}/{partition}_features.csv", "wt") as f:
        #         writer = csv.writer(f)
        #         keys = list(sorted(features[0].keys()))
        #         writer.writerow(keys)
        #         for entry in features:
        #             writer.writerow([entry[f] for f in keys])
        features = filter_rare(dataset)
        print(len(features["train"]))
        job.status_id = JobStatusEnum.FINISHED.value
        job.save()
    except Exception as ex:
        raise ex
        print(ex)
        job.status_id = JobStatusEnum.ERROR.value
        job.save()

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
            

