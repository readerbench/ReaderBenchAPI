import csv
import json
import os
import random
from typing import Dict

from joblib import Parallel, delayed

from services.enums import JobStatusEnum
from services.models import Dataset, Job
from rb import Lang, Document
from rb.complexity.complexity_index import compute_indices
from rb.cna.cna_graph import CnaGraph
from rb.similarity.vector_model_factory import create_vector_model, VectorModelType
from rb.similarity.transformers_encoder import TransformersEncoder

from services.parallel import build_features


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


    
def process(job: Job):
    job.status_id = JobStatusEnum.IN_PROGRESS.value
    job.save()
    try:
        params = json.loads(job.params)
        dataset = Dataset.objects.get(id=params["dataset_id"])
        root = f"data/datasets/{dataset.id}"
        lang = Lang[dataset.lang.label]
        split(dataset)
        for partition in ["train", "val", "test"]:
            features = Parallel(n_jobs=8, prefer="processes")( \
                delayed(build_features)(row[0], lang) \
                for row in generator(dataset, partition))
            with open(f"{root}/{partition}_features.csv", "wt") as f:
                writer = csv.writer(f)
                keys = list(sorted(features[0].keys()))
                writer.writerow(keys)
                for entry in features:
                    writer.writerow([entry[f] for f in keys])

        job.status_id = JobStatusEnum.FINISHED.value
        job.save()
    except Exception as ex:
        raise ex
        print(ex)
        job.status_id = JobStatusEnum.ERROR.value
        job.save()
    