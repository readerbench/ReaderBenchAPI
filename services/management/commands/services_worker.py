import json
import multiprocessing
import os
import time
from datetime import datetime

from django.core.management.base import BaseCommand

from services.enums import JobStatusEnum, JobTypeEnum
from services.models import Job, Language

def get_sentiment(text):
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("readerbench/ro-sentiment")
    model = TFAutoModelForSequenceClassification.from_pretrained("readerbench/ro-sentiment", from_pt=True)
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(**inputs).logits
    scores = tf.nn.softmax(logits)[0].numpy().tolist()
    return {
        "Negative": scores[0],
        "Positive": scores[1]
    }

def offensive_classification(text):
    import tensorflow as tf
    from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained("readerbench/ro-offense")
    model = TFAutoModelForSequenceClassification.from_pretrained("readerbench/ro-offense", from_pt=True)
    inputs = tokenizer(text, return_tensors="tf")
    logits = model(**inputs).logits
    scores = tf.nn.softmax(logits)[0].numpy().tolist()
    return {
        "None": scores[0],
        "Profanity": scores[1],
        "Insults": scores[2],
        "Abuse": scores[3]
    }

def generate_answers_wrapper(text):
    from services.qgen.answer_generation import generate_answers
    return generate_answers(text)

def generate_distractors_wrapper(text, answers):
    from services.qgen.distractors_gen import generate_distractors
    return generate_distractors(text, answers)

def restore_diacritics(text):
    from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration
    model = DiacriticsRestoration()
    return model.process_string(text)

def extract_keywords(text, lang):
    from keybert import KeyBERT
    from transformers.pipelines import pipeline
    lang = Language.objects.filter(id=lang).get().label.lower()
    if lang == "en":
        model = "sentence-transformers/all-MiniLM-L6-v2"
    elif lang == "fr":
        model = "camembert-base"
    elif lang == "ro":
        model = "readerbench/RoBERT-base"
    elif lang == "pt":
        model = "neuralmind/bert-base-portuguese-cased"
    else:
        return None
    hf_model = pipeline("feature-extraction", model=model)
    kw_model = KeyBERT(model=hf_model)
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), use_maxsum=True, nr_candidates=20, top_n=5)
    return [keyword for keyword, score in keywords if score > 0.4]

def job_wrapper(function, queue, **kwargs):
    import torch
    import tensorflow as tf
    result_obj = queue.get()
    try:
        result_obj["result"] = function(**kwargs)
    except Exception as ex:
        result_obj["result"] = None
    queue.put(result_obj)

def run_job(queue, function, params):
    queue.put({"result": None})
    p = multiprocessing.Process(target=job_wrapper, args=[function, queue], kwargs=params)
    p.start()
    p.join()
    return queue.get()["result"]

class Command(BaseCommand):
    help = 'Runs services jobs'

    def handle(self, *args, **options):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        job_types = [
            JobTypeEnum.CSCL.value, JobTypeEnum.OFFENSIVE.value, JobTypeEnum.SENTIMENT.value, JobTypeEnum.DIACRITICS.value,
            JobTypeEnum.INDICES.value, JobTypeEnum.ANSWER_GEN.value, JobTypeEnum.TEST_GEN.value, JobTypeEnum.KEYWORDS.value,
        ]
        queue = multiprocessing.Queue()
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
                    result = run_job(queue, generate_answers_wrapper, params)
                elif job.type_id == JobTypeEnum.TEST_GEN.value:
                    result = run_job(queue, generate_distractors_wrapper, params)
                elif job.type_id == JobTypeEnum.DIACRITICS.value:
                    result = run_job(queue, restore_diacritics, params)
                elif job.type_id == JobTypeEnum.SENTIMENT.value:
                    result = run_job(queue, get_sentiment, params)
                elif job.type_id == JobTypeEnum.OFFENSIVE.value:
                    result = run_job(queue, offensive_classification, params)
                elif job.type_id == JobTypeEnum.KEYWORDS.value:
                    result = run_job(queue, extract_keywords, params)
                if result is None:
                    raise Exception()
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
            

