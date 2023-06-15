import re
import string
from typing import Dict, List
from rb.parser.spacy_parser import SpacyParser
from rb import Lang
import tensorflow as tf
from transformers import AutoTokenizer, TFT5ForConditionalGeneration
def spacy_gen(text: str) -> List[Dict]:
    doc = SpacyParser.get_instance().get_model(Lang.EN)(text)
    result = []
    for ent in doc.ents:
        result.append({
            "start": ent.start_char,
            "end": ent.end_char,
            "type": "NER",
            "text": ent.text,
        })
    return result

def normalize(text: str) -> str:
    text = text.strip()
    if text[-1] in string.punctuation:
        return normalize(text[:-1])
    return text

def oracle_gen(text: str) -> List[Dict]:
    tf.config.set_visible_devices([], 'GPU')
    tokenizer = AutoTokenizer.from_pretrained('readerbench/AG-Flan-T5-large')
    model = TFT5ForConditionalGeneration.from_pretrained('readerbench/AG-Flan-T5-large')
    prompt = f"Select an answer from the context that can be used to generate a question:\nContext: {text}"
    input_ids = tokenizer(prompt, return_tensors="tf").input_ids
    unique = set()
    prediction = model.generate(input_ids=input_ids, max_new_tokens=32,  
                                num_return_sequences=1, penalty_alpha=0.6, top_k=8)
    for prediction in tokenizer.batch_decode(prediction, skip_special_tokens=True):
        unique.add(normalize(prediction))
    prediction = model.generate(input_ids=input_ids, max_new_tokens=32, num_beams=8,
                                num_return_sequences=2)
    for prediction in tokenizer.batch_decode(prediction, skip_special_tokens=True):
        unique.add(normalize(prediction))
    prediction = model.generate(input_ids=input_ids, max_new_tokens=32, top_p=0.95, top_k=0, 
                                do_sample=True, num_return_sequences=16)
    for prediction in tokenizer.batch_decode(prediction, skip_special_tokens=True):
        unique.add(normalize(prediction))
    result = []
    for answer in unique:
        for m in re.finditer(answer, text):
            result.append({
                "start": m.start(),
                "end": m.end(),
                "type": "Oracle",
                "text": answer,
            })
    return result