import csv
import json
import logging
import os
import time

from django.core.management.base import BaseCommand
from joblib import Parallel, delayed
from rb import Lang

from pipeline.parallel import build_features
from pipeline.preprocessing import filter_rare, generator, get_labels, get_tasks, remove_colinear, split
from pipeline.task import Task
from pipeline.tuning import hyperparameter_search
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
        with open(f"{root}/targets.csv", "rt") as f:
            reader = csv.reader(f)
            header = next(reader)
            task_names = header[1:]
        split(dataset)
        for partition in ["train", "val", "test"]:
            with Parallel(n_jobs=8, prefer="processes", verbose=100) as parallel:
                features = parallel( \
                    delayed(build_features)(row[0], lang) \
                    for row in generator(dataset, partition))
            with open(f"{root}/{partition}_features.csv", "wt") as f:
                writer = csv.writer(f)
                keys = list(sorted(features[0].keys()))
                writer.writerow(keys)
                for entry in features:
                    writer.writerow([entry[f] for f in keys])
        features = filter_rare(dataset)
        all_labels = get_labels(dataset)
        tasks = get_tasks(all_labels)
        for name, task in zip(task_names, tasks):
            task.name = name
        labels = [[] for _ in tasks]
        for row in generator(dataset, "train"):
            for i, value in enumerate(row[1:]):
                labels[i].append(value)
        for task, targets in zip(tasks, labels):
            task.features = remove_colinear(features["train"], task.convert_targets(targets))
        for i, task in enumerate(tasks):
            task.save(f"data/datasets/{dataset.id}/task_{i}.json")
        # tasks = []
        # for i in range(len(task_names)):
        #     with open(f"data/datasets/{dataset.id}/task_{i}.json", "rt") as f:
        #         obj = json.load(f)
        #         task = Task(obj=obj)
        #         tasks.append(task)
        hyperparameter_search(dataset, job, tasks, features)
        job.status_id = JobStatusEnum.FINISHED.value
        job.save()
    except KeyboardInterrupt:
        raise
    except Exception as ex:
        print(ex)
        job.status_id = JobStatusEnum.ERROR.value
        job.save()

class Command(BaseCommand):
    help = 'Runs pipeline jobs'

    def handle(self, *args, **options):
        while True:
            job = Job.objects.filter(status_id=JobStatusEnum.PENDING.value, type_id=JobTypeEnum.PIPELINE.value).first()
            if job is None:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    raise
                continue
            print(f"Starting pipeline job {job.id}...")
            process(job)
            print(f"Pipeline job {job.id} finished")
            

