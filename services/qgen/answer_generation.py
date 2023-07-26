import re
import string
from typing import Dict, List

import tensorflow as tf
from cleantext import clean
from nltk import sent_tokenize
from rb import Lang
from rb.parser.spacy_parser import SpacyParser
from transformers import AutoTokenizer, TFT5ForConditionalGeneration

from services.qgen import environment
from services.qgen.dqn import DQAgent, get_largest_indexes


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
    # tf.config.set_visible_devices([], 'GPU')
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

def rl_gen(text) -> List[Dict]:
    sentences = sent_tokenize(text)
    index = 0
    sent_start = {}
    for i, sent in enumerate(sentences):
        if not text[index:].startswith(sent):
            index += 1
        sent_start[i] = index
        index += len(sent)    
    agent = DQAgent()
    agent.load('models/model_dq_val_0')
    try:
        env = environment.MyEnvironment(text)
        env.assign_parents(env.initial_state)
    except Exception as e:
        raise

    _, dists, _ = agent.act(env.current_state)
    best_indexes = get_largest_indexes(dists, len(sentences))

    all_answers = []

    for i in range(len(sentences)):
        env.current_state = env.initial_state
        index = best_indexes[i]
        done = False
        while not done:
            action, dists, _ = agent.act(env.current_state, index)
            print("-------")
            state: environment.Node = env.current_state
            print(f"Current state: {state.text}")
            done = env.step_next_state(action)
            if done == True:
                all_answers.append({
                    "start": state.start_char_idx + sent_start[index],
                    "end": state.end_char_idx + sent_start[index],
                    "type": "RL",
                    "text": state.text,
                })
                break
    return all_answers

def generate_answers(text: str) -> List[Dict]:
    text = clean(text, fix_unicode=True, to_ascii=True, lower=False, no_urls=True, no_emails=True, no_phone_numbers=True)
    result = []
    result += spacy_gen(text)
    result += oracle_gen(text)
    result += rl_gen(text)
    return {
        "text": text,
        "answers": result
    }