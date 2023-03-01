import datetime
import json
from typing import Dict, List

import numpy as np
import tensorflow as tf
from pipeline.enums import ModelTypeEnum
from pipeline.models import Model
from pipeline.preprocessing import generator
from pipeline.task import Task
from rb import Lang
from services.models import Dataset, Job



def hyperparameter_search(dataset: Dataset, job: Job, tasks: List[Task], features: Dict[str, Dict[str, List[float]]]):
    texts = {}
    targets = {}
    for partition in ["train", "val", "test"]:
        texts[partition] = []
        targets[partition] = [[] for task in tasks]
        for row in generator(dataset, partition):
            texts[partition].append(row[0])
            for i, val in enumerate(row[1:]):
                targets[partition][i].append(val)
        for i, task in enumerate(tasks):
            targets[partition][i] = task.convert_targets(targets[partition][i])
    lang = Lang[dataset.lang.label]
    for predictor_type in ModelTypeEnum:
        predictor = predictor_type.predictor()(lang, tasks)
        predictor.load_data(texts, features, targets)
        best_result = predictor.search_config()
        config = predictor.process_result(best_result)
        metrics = predictor.train(config, validation=False)
        obj = Model()
        obj.dataset = dataset
        obj.job = job
        obj.type_id = predictor_type.value
        obj.params = json.dumps(config)
        obj.metrics = json.dumps(metrics)
        obj.save()
        predictor.save(obj)