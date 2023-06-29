from transformers import AutoTokenizer, TFT5ForConditionalGeneration, TFRobertaForMaskedLM
import tensorflow as tf
from tqdm import tqdm
from sentence_transformers import CrossEncoder
tf.config.set_visible_devices([], 'GPU')
    
t5_tokenizer_qall = AutoTokenizer.from_pretrained('readerbench/QAll-Flan-xl')
t5_model_qall = TFT5ForConditionalGeneration.from_pretrained('readerbench/QAll-Flan-xl')
deberta_model_nli = CrossEncoder('cross-encoder/nli-deberta-v3-base')
roberta_tokenizer_mlm = AutoTokenizer.from_pretrained("roberta-base")
roberta_model_mlm = TFRobertaForMaskedLM.from_pretrained("roberta-base")
t5_model_xl = TFT5ForConditionalGeneration.from_pretrained('google/t5-v1_1-xl')
t5_tokenizer_xl = AutoTokenizer.from_pretrained('google/t5-v1_1-xl')

def generate_distractors_answer(context, question):
    list_string = [f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"]
    
    decodes_list = []
    batch_size=8
    for i in tqdm(range(0, len(list_string), batch_size)):

        end_interval = min(i+batch_size, len(list_string))

        tokens = t5_tokenizer_qall(list_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        #model_prediction = t5_model_qall.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], do_sample=True, max_new_tokens=512, penalty_alpha=0.6, top_k=8, num_return_sequences=100)
        #model_prediction = t5_model_qall.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], do_sample=True, num_return_sequences=100)
        model_prediction = t5_model_qall.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'], do_sample=True, num_beams=30, num_return_sequences=30, max_new_tokens=32)
        decodes = t5_tokenizer_qall.batch_decode(model_prediction, skip_special_tokens=True)
        decodes_list += decodes
    return decodes_list

def generate_mlm_distractors(sentence, correct_answer):
    list_string = [sentence.replace('**blank**', f"<extra_id_0> (or {correct_answer})")]
    t5_tokenizer_xl.eos_token_id = t5_tokenizer_xl.vocab['<extra_id_1>']
    t5_tokenizer_xl.pad_token_id = t5_tokenizer_xl.eos_token_id
    tokens = t5_tokenizer_xl(list_string, return_tensors='tf', max_length=512, padding='max_length', truncation=True)
    model_prediction = t5_model_xl.generate(return_dict_in_generate=True, output_scores=True, input_ids=tokens['input_ids'], \
                                            attention_mask=tokens['attention_mask'], do_sample=True, num_return_sequences=16, \
                                            eos_token_id=t5_tokenizer_xl.vocab['<extra_id_1>'], max_new_tokens=32)
    decodes = t5_tokenizer_xl.batch_decode(model_prediction.sequences, skip_special_tokens=True)
    return decodes

def get_fitness_loss_no_finetune(sentence, distractors, correct_answer):
    t5_tokenizer_xl.eos_token_id = t5_tokenizer_xl.vocab['<extra_id_1>']
    t5_tokenizer_xl.pad_token_id = t5_tokenizer_xl.eos_token_id
    input_string = [sentence.replace('**blank**', f"<extra_id_0>") for _ in distractors]
    labels_string = [f"<extra_id_0>{d}<extra_id_1>" for d in distractors]

    batch_size=8
    losses = []

    for i in range(0, len(input_string), batch_size):
        end_interval = min(i+batch_size, len(input_string))
        input_tokens = t5_tokenizer_xl(input_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels_tokens = t5_tokenizer_xl(labels_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels = labels_tokens.input_ids

        outputs = t5_model_xl(input_ids=input_tokens.input_ids, attention_mask=input_tokens.attention_mask, labels=labels)
        logits = outputs.logits
        num_classes = logits.shape[2]
        labels_one_hot = tf.one_hot(labels, depth=num_classes)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        for logit, label_oh, label in zip(logits, labels_one_hot, labels):
            num_elements = tf.math.count_nonzero(label != t5_tokenizer_xl.pad_token_id)
            good_label_oh = label_oh[:num_elements]
            good_logit = logit[:num_elements]
            loss = loss_fn(
                tf.reshape(good_label_oh, [1] + good_label_oh.shape),
                tf.reshape(good_logit, [1] + good_logit.shape), 
            )
            losses.append(float(loss))
    return losses


def get_entailment(first_sentences, second_sentences):
    nli_list = [(f, s) for f, s in zip(first_sentences, second_sentences)]
    scores = deberta_model_nli.predict(nli_list)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels

def get_entailment_score(first_sentences, second_sentences):
    nli_list = [(f, s) for f, s in zip(first_sentences, second_sentences)]
    scores = deberta_model_nli.predict(nli_list)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [score.tolist() for score in scores]
    #print(labels)
    return labels

def get_answer_loss(context, question, answers):
    input_string = [f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}" for _ in answers]
    labels_string = [answer for answer in answers]

    batch_size=8
    losses = []

    for i in tqdm(range(0, len(input_string), batch_size)):
        end_interval = min(i+batch_size, len(input_string))
        input_tokens = t5_tokenizer_qall(input_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels_tokens = t5_tokenizer_qall(labels_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels = labels_tokens.input_ids

        outputs = t5_model_qall(input_ids=input_tokens.input_ids, attention_mask=input_tokens.attention_mask, labels=labels)
        logits = outputs.logits
        num_classes = logits.shape[2]
        labels_one_hot = tf.one_hot(labels, depth=num_classes)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        for logit, label_oh, label in zip(logits, labels_one_hot, labels):
            num_elements = tf.math.count_nonzero(label != t5_tokenizer_qall.pad_token_id)
            good_label_oh = label_oh[:num_elements]
            good_logit = logit[:num_elements]
            loss = loss_fn(
                tf.reshape(good_label_oh, [1] + good_label_oh.shape),
                tf.reshape(good_logit, [1] + good_logit.shape), 
            )
            losses.append(float(loss))
    return losses

def get_mlm_loss(entity, possibilities):
    input_string = [f"{entity} is a {roberta_tokenizer_mlm.mask_token}" for possibility in possibilities]
    labels_string = [possibility for possibility in possibilities]

    batch_size=12
    losses = []

    for i in tqdm(range(0, len(input_string), batch_size)):
        end_interval = min(i+batch_size, len(input_string))
        input_tokens = roberta_tokenizer_mlm(input_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels_tokens = roberta_tokenizer_mlm(labels_string[i:end_interval], return_tensors='tf', max_length=512, padding='max_length', truncation=True)
        labels = labels_tokens.input_ids

        outputs = roberta_model_mlm(input_ids=input_tokens.input_ids, attention_mask=input_tokens.attention_mask, labels=labels)
        logits = outputs.logits
        num_classes = logits.shape[2]
        labels_one_hot = tf.one_hot(labels, depth=num_classes)
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        for logit, label_oh, label in zip(logits, labels_one_hot, labels):
            num_elements = tf.math.count_nonzero(label != roberta_tokenizer_mlm.pad_token_id)
            good_label_oh = label_oh[:num_elements]
            good_logit = logit[:num_elements]
            loss = loss_fn(
                tf.reshape(good_label_oh, [1] + good_label_oh.shape),
                tf.reshape(good_logit, [1] + good_logit.shape), 
            )
            losses.append(float(loss))
    return losses
