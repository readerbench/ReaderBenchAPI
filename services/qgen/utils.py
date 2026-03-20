import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5ForConditionalGeneration, RobertaForMaskedLM
from tqdm import tqdm
from sentence_transformers import CrossEncoder
from sense2vec import Sense2Vec


def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t5_tokenizer_qall = AutoTokenizer.from_pretrained('readerbench/QAll-Flan-large')
    t5_model_qall = T5ForConditionalGeneration.from_pretrained('readerbench/QAll-Flan-large').to(device)
    deberta_model_nli = CrossEncoder('cross-encoder/nli-deberta-v3-base')

    t5_model_xl = T5ForConditionalGeneration.from_pretrained('google/t5-v1_1-large').to("cpu")
    t5_tokenizer_xl = AutoTokenizer.from_pretrained('google/t5-v1_1-large')

    return {
        "qall_tokenizer": t5_tokenizer_qall,
        "qall_model": t5_model_qall,
        "nli_model": deberta_model_nli,
        "t5_tokenizer": t5_tokenizer_xl,
        "t5_model": t5_model_xl,
        "s2v": Sense2Vec().from_disk('models/s2v_reddit_2019_lg')
    }


def generate_distractors_answer(context, question, models):
    list_string = [f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}"]

    decodes_list = []
    batch_size = 8
    device = next(models["qall_model"].parameters()).device
    for i in tqdm(range(0, len(list_string), batch_size)):
        end_interval = min(i + batch_size, len(list_string))
        tokens = models["qall_tokenizer"](
            list_string[i:end_interval], return_tensors='pt', max_length=512,
            padding='max_length', truncation=True
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        model_prediction = models["qall_model"].generate(
            input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
            do_sample=True, num_beams=30, num_return_sequences=30, max_new_tokens=32
        )
        decodes = models["qall_tokenizer"].batch_decode(model_prediction, skip_special_tokens=True)
        decodes_list += decodes
    return decodes_list


def generate_mlm_distractors(sentence, correct_answer, models):
    list_string = [sentence.replace('**blank**', f"<extra_id_0> (or {correct_answer})")]
    models["t5_tokenizer"].eos_token_id = models["t5_tokenizer"].vocab['<extra_id_1>']
    models["t5_tokenizer"].pad_token_id = models["t5_tokenizer"].eos_token_id
    device = next(models["t5_model"].parameters()).device
    tokens = models["t5_tokenizer"](list_string, return_tensors='pt', max_length=512, truncation=True)
    tokens = {k: v.to(device) for k, v in tokens.items()}
    model_prediction = models["t5_model"].generate(
        return_dict_in_generate=True, output_scores=True,
        input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'],
        do_sample=True, num_return_sequences=32,
        eos_token_id=models["t5_tokenizer"].vocab['<extra_id_1>'], max_new_tokens=32
    )
    decodes = models["t5_tokenizer"].batch_decode(model_prediction.sequences, skip_special_tokens=True)
    return decodes


def get_fitness_loss_no_finetune(sentence, distractors, correct_answer, models):
    tokenizer = models["t5_tokenizer"]
    tokenizer.eos_token_id = tokenizer.vocab['<extra_id_1>']
    tokenizer.pad_token_id = tokenizer.eos_token_id
    input_string = [sentence.replace('**blank**', f"<extra_id_0>") for _ in distractors]
    labels_string = [f"<extra_id_0>{d}<extra_id_1>" for d in distractors]

    batch_size = 8
    losses = []
    device = next(models["t5_model"].parameters()).device

    for i in range(0, len(input_string), batch_size):
        end_interval = min(i + batch_size, len(input_string))
        input_tokens = tokenizer(
            input_string[i:end_interval], return_tensors='pt', max_length=512, padding=True, truncation=True
        )
        labels_tokens = tokenizer(
            labels_string[i:end_interval], return_tensors='pt', max_length=512, padding=True, truncation=True
        )
        input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
        labels = labels_tokens['input_ids'].to(device)

        with torch.no_grad():
            outputs = models["t5_model"](
                input_ids=input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                labels=labels
            )
        logits = outputs.logits  # (batch, seq, vocab)
        num_classes = logits.shape[2]

        for logit, label in zip(logits, labels):
            non_pad_mask = label != tokenizer.pad_token_id
            num_elements = non_pad_mask.sum().item()
            good_logit = logit[:num_elements]  # (num_elements, vocab)
            good_label = label[:num_elements]  # (num_elements,)
            good_label_oh = F.one_hot(good_label, num_classes=num_classes).float()
            loss = F.cross_entropy(good_logit, good_label_oh)
            losses.append(float(loss))
    return losses


def get_entailment(first_sentences, second_sentences, models):
    nli_list = [(f, s) for f, s in zip(first_sentences, second_sentences)]
    scores = models["nli_model"].predict(nli_list)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]
    return labels


def get_entailment_score(first_sentences, second_sentences, models):
    nli_list = [(f, s) for f, s in zip(first_sentences, second_sentences)]
    scores = models["nli_model"].predict(nli_list)
    labels = [score.tolist() for score in scores]
    return labels


def get_answer_loss(context, question, answers, models):
    input_string = [f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}" for _ in answers]
    labels_string = [answer for answer in answers]

    batch_size = 4
    losses = []
    device = next(models["qall_model"].parameters()).device

    for i in tqdm(range(0, len(input_string), batch_size)):
        end_interval = min(i + batch_size, len(input_string))
        input_tokens = models["qall_tokenizer"](
            input_string[i:end_interval], return_tensors='pt', max_length=512, padding=True, truncation=True
        )
        labels_tokens = models["qall_tokenizer"](
            labels_string[i:end_interval], return_tensors='pt', max_length=512, padding=True, truncation=True
        )
        input_tokens = {k: v.to(device) for k, v in input_tokens.items()}
        labels = labels_tokens['input_ids'].to(device)

        with torch.no_grad():
            outputs = models["qall_model"](
                input_ids=input_tokens['input_ids'],
                attention_mask=input_tokens['attention_mask'],
                labels=labels
            )
        logits = outputs.logits
        num_classes = logits.shape[2]

        for logit, label in zip(logits, labels):
            non_pad_mask = label != models["qall_tokenizer"].pad_token_id
            num_elements = non_pad_mask.sum().item()
            good_logit = logit[:num_elements]
            good_label = label[:num_elements]
            good_label_oh = F.one_hot(good_label, num_classes=num_classes).float()
            loss = F.cross_entropy(good_logit, good_label_oh)
            losses.append(float(loss))
    return losses
