from services.qgen.distractors_gen import get_distractors_conceptnet, get_distractors_dbpedia2, get_distractors_dbpedia3, get_distractors_dbpedia4, get_distractors_sense2vec, get_distractors_wordnet, locate_dbpedia_uri

import nltk

from services.qgen.utils import generate_mlm_distractors, get_answer_loss, get_entailment


def generate_all_distractors(context, question, answer, answer_containing_sentence, num_distractors=3):

    distractors = []
    blanked_sentence = answer_containing_sentence.replace(answer, '**blank**')
    #distractors += utils.generate_distractors_answer(context.replace(answer_containing_sentence, ''), question)
    distractors += generate_mlm_distractors(context.replace(answer_containing_sentence, blanked_sentence), answer)
    try:
        input_uri = locate_dbpedia_uri(answer, blanked_sentence)
    except:
        pass
    try:
        distractors += get_distractors_dbpedia2(input_uri)
    except:
        pass
    try:
        distractors += get_distractors_dbpedia3(input_uri)
    except:
        pass
    try:
        distractors += get_distractors_dbpedia4(input_uri)
    except:
        pass
    try:
        distractors += get_distractors_wordnet(answer)
    except:
        pass
    try:
        distractors += get_distractors_conceptnet(answer)
    except:
        pass
    try:
        distractors += get_distractors_sense2vec(answer)
    except:
        pass

    answer_pos_tag = nltk.pos_tag([answer], tagset='universal')[0][1]
    distractors_pos_tags = nltk.pos_tag(distractors, tagset='universal')
    distractors = [dist for dist, pos in distractors_pos_tags if pos == answer_pos_tag]

    # Filter entailment
    first_sentences = [answer_containing_sentence for _ in distractors]
    second_sentences = [answer_containing_sentence.replace(answer, d) for d in distractors]

    nli_labels = get_entailment(first_sentences, second_sentences)

    distractors_to_keep = [d for d, l in zip(distractors, nli_labels) if l == 'contradiction']
    distractors_to_discard = [d for d, l in zip(distractors, nli_labels) if l == 'entailment']

    qa_losses = get_answer_loss(context, question, distractors_to_keep)

    distractors_losses = zip(distractors_to_keep, qa_losses)
    distractors_losses = sorted(distractors_losses, key=lambda x: x[1])

    final_distractors = [distractors_losses[0]]
    i = 1
    while len(final_distractors) < num_distractors or i == len(distractors_losses):
        first_sentences = [answer_containing_sentence.replace(answer, d[0]) for d in final_distractors]
        second_sentences = [answer_containing_sentence.replace(answer, distractors_losses[i][0]) for _ in final_distractors]
        nli_labels = get_entailment(first_sentences, second_sentences)
        nli_labels += get_entailment(second_sentences, first_sentences)
        if 'entailment' not in nli_labels:
            final_distractors.append(distractors_losses[i])
        i += 1

    return final_distractors
    