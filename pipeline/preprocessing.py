import csv
from heapq import heapify, heappop
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List

from scipy.stats import f_oneway, pearsonr

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

def generator(dataset: Dataset, partition: str):
    root = f"data/datasets/{dataset.id}"
    with open(f"{root}/{partition}.txt", "rt") as f:
        ids = {int(line) for line in f.readlines()}
    sep_file = os.path.exists(f"{root}/texts")
    with open(f"{root}/targets.csv", "rt") as f:
        reader = csv.reader(f)
        next(reader)
        for i, row in enumerate(reader):
            if i not in ids:
                continue
            if sep_file:
                with open(f"{root}/texts/{row[0]}", "rt") as text_file:
                    text = text_file.read()
            else:
                text = row[0]
            yield (text,) + tuple(row[1:])

def filter_rare(dataset: Dataset):
    root = f"data/datasets/{dataset.id}"
    with open(f"{root}/train_features.csv", "rt") as f:
        reader = csv.reader(f)
        keys = next(reader)
        all_values = {key: [] for key in keys}
        for row in reader:
            for key, value in zip(keys, row):
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
                        result[partition][key].append(value)
    return result

def convert_labels(labels: List[List[str]]) -> List[Task]:
    values = zip(*labels)
    tasks = []
    for targets in values:
        if all(is_double(target) for target in targets):
            tasks.append(Task(TargetType.FLOAT, targets))
        elif all(is_int(target) for target in targets):
            tasks.append(Task(TargetType.INT, targets))
        else:
            tasks.append(Task(TargetType.STR, targets))
    return tasks

def correlation_with_targets(feature: str, docs: List[Dict[str, float]], labels: List[float]) -> float:
    values = [doc[feature] for doc in docs]
    corr, p = pearsonr(values, labels)
    return abs(corr)
   

def remove_colinear(features: List[str], docs: List[Dict[str, float]], labels: List[float]) -> List[str]:
    heap = []
    for i, a in enumerate(features[:-1]):
        for j in range(i+1, len(features)):
            b = features[j]
            values_a = [doc[a] for doc in docs]
            values_b = [doc[b] for doc in docs]
            corr, p = pearsonr(values_a, values_b)
            if math.isnan(corr):
                continue
            heap.append((-corr, i, j))
    heapify(heap)
    
    correlations = [
        correlation_with_targets(feature, docs, labels) 
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
    return [
        feature
        for feature, m in zip(features, mask)
        if mask
    ]


    