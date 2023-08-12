import csv
from heapq import heapify, heappop
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List

from scipy.stats import f_oneway, pearsonr
from pipeline.task import TargetType, Task, is_double, is_int

from services.models import Dataset, Job


def split(dataset: Dataset):
    ids = list(range(dataset.num_rows))
    random.shuffle(ids)
    train_len = int(0.6 * len(ids))
    test_len = int(0.2 * len(ids))
    root = f"data/datasets/{dataset.id}"
    with open(f"{root}/train.txt", "wt") as f:
        for idx in ids[:train_len]:
            f.write(str(idx) + "\n")
    with open(f"{root}/val.txt", "wt") as f:
        for idx in ids[train_len:-test_len]:
            f.write(str(idx) + "\n")
    with open(f"{root}/test.txt", "wt") as f:
        for idx in ids[-test_len:]:
            f.write(str(idx) + "\n")

def get_labels(dataset: Dataset) -> List[List[str]]:
    root = f"data/datasets/{dataset.id}"
    result = []
    with open(f"{root}/targets.csv", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            result.append(row[1:])
    return result
    

def generator(dataset: Dataset, partition: str = None):
    root = f"data/datasets/{dataset.id}"
    if partition:
        with open(f"{root}/{partition}.txt", "rt") as f:
            ids = {int(line) for line in f.readlines()}
    sep_file = os.path.exists(f"{root}/texts")
    with open(f"{root}/targets.csv", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if partition and i not in ids:
                continue
            if sep_file:
                with open(f"{root}/texts/{row[0]}", "rt") as text_file:
                    text = text_file.read()
            else:
                text = row[0]
            yield (text,) + tuple(row[1:])
            

def filter_rare(dataset: Dataset) -> Dict[str, Dict[str, List[float]]]:
    root = f"data/datasets/{dataset.id}"
    with open(f"{root}/train_features.csv", "rt") as f:
        reader = csv.reader(f)
        keys = next(reader)
        all_values = {key: [] for key in keys}
        for row in reader:
            for key, value in zip(keys, row):
                if value is not None and value != "":
                    value = float(value)
                else:
                    value = 0.
                all_values[key].append(value)
    features = []
    for key, values in all_values.items():
        counter = Counter(values)
        if counter.most_common(1)[0][1] < 0.2 * len(values):
            features.append(key)
    result = {
        "train": {
            f: all_values[f]
            for f in features
        }
    }
    features = set(features)
    for partition in ["val", "test"]:
        result[partition] = {
            f: []
            for f in features
        }
        with open(f"{root}/{partition}_features.csv", "rt") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                for key, value in zip(keys, row):
                    if key in features:
                        if value is not None and value != "":
                            value = float(value)
                        else:
                            value = 0.
                        result[partition][key].append(value)
    return result

def get_tasks(labels: List[List[str]]) -> List[Task]:
    values = zip(*labels)
    tasks = []
    for targets in values:
        if all(is_int(target) for target in targets):
            tasks.append(Task(TargetType.INT, targets))
        elif all(is_double(target) for target in targets):
            tasks.append(Task(TargetType.FLOAT, targets))
        else:
            tasks.append(Task(TargetType.STR, targets))
    return tasks

def correlation_with_targets(values: List[float], labels: List[float]) -> float:
    corr, p = pearsonr(values, labels)
    return abs(corr)
   

def remove_colinear(values: Dict[str, List[float]], labels: List[float]) -> List[str]:
    heap = []
    features = list(values.keys())
    for i, a in enumerate(features[:-1]):
        for j in range(i+1, len(features)):
            b = features[j]
            corr, p = pearsonr(values[a], values[b])
            if math.isnan(corr):
                continue
            heap.append((-corr, i, j))
    heapify(heap)
    
    correlations = [
        correlation_with_targets(values[feature], labels) 
        for feature in features
    ]
    mask = [True] * len(features)
    while len(heap) > 0:
        inv_corr, i, j = heappop(heap)
        if not mask[i] or not mask[j]:
            continue
        if inv_corr < -0.9:
            if correlations[i] > correlations[j]:
                mask[j] = False
            else:
                mask[i] = False
    result = [
        (feature, c)
        for feature, m, c in zip(features, mask, correlations)
        if m
    ]
    result.sort(key=lambda x: x[1], reverse=True)
    return [feature for feature, _ in result[:100]]


    