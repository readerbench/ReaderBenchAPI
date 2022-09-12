from rb.parser.spacy_parser import SpacyParser
from services.subject_predicate.solver import *
from services.subject_predicate.subiect import Subiect
from services.subject_predicate.predicat import Predicat
from services.subject_predicate.z_sentence import get_sentences




def language_correct(text, language, dic_path, aff_path):
    import hunspell
    en_hunspell = hunspell.HunSpell(dic_path, aff_path)
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if len(p) > 0]
    corrections, split_text = [], []
    
    spacyInstance = SpacyParser.get_instance()

    for p_index, text in enumerate(paragraphs):
        doc = spacyInstance.parse(text, language)
        p_list = []
        for i, token in enumerate(doc):
            p_list.append(token.text)
            if token.is_punct == False and en_hunspell.spell(token.text) == False:
                corrections.append({
                    'mistake': "Spellchecking",
                    'index': [[p_index, i]],
                    "suggestions": sort_suggestions_by_similarity(doc, en_hunspell.suggest(token.text), spacyInstance,
                                                                  language)
                })
        split_text.append(p_list)

    res = {'corrections': corrections, 'split_text': split_text}
    return res


def ro_language_correct(text, ro_model):
    paragraphs = text.split('\n')
    paragraphs = [p for p in paragraphs if len(p) > 0]
    corrections, split_text = [], []
    for p_index, text in enumerate(paragraphs):
        sent_index = 0
        doc = ro_model(text)
        sents = list(doc.sents)
        split_text.append([str(x) for x in sents])
        for sent in sents:
            try:
                errors = subiect_predicat_check(sent, ro_model)
                for error in errors:
                    corrections.append({
                        'mistake': error,
                        'index_paragraph': p_index,
                        'index_sentence': sent_index,
                        'components': f'Subiect: {Subiect(sent)}, Predicat: {Predicat(sent)}'
                    })
            except:
                pass
            sent_index += 1
    res = {'corrections': corrections, 'split_text': split_text}
    return res


def subiect_predicat_check(doc, ro_model):
    clauses = get_sentences(doc)
    errors = []
    for clause in clauses:
        print(clause)
        errors.extend(text_check(ro_model(clause), ro_model))
    return errors


def sort_suggestions_by_similarity(doc, suggestions, parser, lang):
    suggestion_score_list = []
    for suggestion in suggestions:
        s = parser.parse(suggestion, lang)
        suggestion_score_list.append((suggestion, doc.similarity(s)))
    suggestion_score_list.sort(key=lambda x: x[1], reverse=True)
    return [x[0] for x in suggestion_score_list]


if __name__ == "__main__":
    text = "Bunica lucrez de acasă."
    ro_model = spacy.load('ro_core_news_lg')
    doc = ro_model(text)
    print(subiect_predicat_check(doc, ro_model))