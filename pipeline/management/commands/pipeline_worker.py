import csv
import json
import logging
import os
import time
from datetime import datetime

from django.core.management.base import BaseCommand
from django.db.models import Q
from joblib import Parallel, delayed
from rb import Lang
import tensorflow as tf

from pipeline.enums import ModelTypeEnum
from pipeline.models import Model
from pipeline.parallel import build_features
from pipeline.preprocessing import (filter_rare, generator, get_labels,
                                    get_tasks, remove_colinear, split)
from pipeline.task import TargetType, Task
from pipeline.tuning import hyperparameter_search
from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Dataset, Job, Language


def process(job: Job):
    job.status_id = JobStatusEnum.IN_PROGRESS.value
    job.save()
    t1 = datetime.now()
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

def predict(job: Job):
    job.status_id = JobStatusEnum.IN_PROGRESS.value
    job.save()
    t1 = datetime.now()
    try:
        params = json.loads(job.params)
        model = Model.objects.get(id=params["model_id"])
        dataset = model.dataset
        tasks = []
        for i in range(dataset.num_cols):
            with open(f"data/datasets/{dataset.id}/task_{i}.json", "rt") as f:
                obj = json.load(f)
                task = Task(obj=obj)
                tasks.append(task)
        lang = Lang[dataset.lang.label]
        predictor = ModelTypeEnum(model.type_id).predictor()(lang, tasks)
        texts = []
        with open(f"data/jobs/{job.id}/input.csv", "rt") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                texts.append(row[0])
        with Parallel(n_jobs=4, prefer="processes", verbose=100) as parallel:
            all_features = parallel( \
                delayed(build_features)(text, lang) \
                for text in texts)
        predictions = predictor.predict(model, texts, all_features)
        with open(f"data/jobs/{job.id}/output.csv", "wt") as f:
            writer = csv.writer(f)
            header = ["Text"]
            for task in tasks:
                if task.type is TargetType.STR:
                    for label in task.classes:
                        header.append(f"{task.name}: {label}")
                else:
                    header.append(task.name)
            writer.writerow(header)
            for i, text in enumerate(texts):
                row = [text]
                for task, pred in zip(tasks, predictions):
                    if task.type is TargetType.STR:
                        row += [float(f) for p in pred[i]]
                    else:
                        row.append(float(pred[i]))
                writer.writerow(row)    
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

class Command(BaseCommand):
    help = 'Runs pipeline jobs'

    def handle(self, *args, **options):
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        gpus = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(gpus[1], 'GPU')
        while True:
            job = Job.objects \
                .filter(status_id=JobStatusEnum.PENDING.value) \
                .filter(Q(type_id=JobTypeEnum.PIPELINE.value) | Q(type_id=JobTypeEnum.PREDICT.value)) \
                .exclude(params="{}") \
                .first()
            if job is None:
                try:
                    time.sleep(1)
                except KeyboardInterrupt:
                    raise
                continue
            print(f"Starting pipeline job {job.id}...")
            if job.type_id == JobTypeEnum.PIPELINE.value:
                process(job)
            else:
                predict(job)
            print(f"Pipeline job {job.id} finished")
            

