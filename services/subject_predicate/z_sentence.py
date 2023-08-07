import spacy


def get_words_in_subtree(word):
    return word.subtree


def get_words_from_intersted_clauses(sentence, clauses=['ROOT', 'conj', 'csubj', 'ccomp', 'advcl', 'acl']):
    clauses_head = []
    with_pass_clauses = clauses + [x + ':pass' for x in clauses]
    for word in sentence:
        if word.dep_ in with_pass_clauses:
            clauses_head.append(word)
    return clauses_head


def check_if_clause_has_predicativly_verb(clause_head):
    if clause_head.dep_ == 'ROOT':
        return True
    if clause_head.tag_[0] == 'V' and len(clause_head.tag_) > 2 and clause_head.tag_[2] != 'g': # in ['i', 's', 'm', 'n', 't']
        return True
    for child in clause_head.children:
        if child.dep_.startswith('cop'):
            return True
    return False


def get_sentences(doc):
    root = None
    for word in doc:
        if word.dep_ == 'ROOT':
            root = word
            break
    
    if not root:
        return None
    
    interested_clauses = get_words_from_intersted_clauses(doc)
    filtered_clauses = [clause_head for clause_head in interested_clauses if check_if_clause_has_predicativly_verb(clause_head)]
    sentences = []
    for sentence_head in filtered_clauses:
        sentences.append(" ".join([str(x) for x in list(sentence_head.subtree)]))
    return sentences
    

if __name__ == '__main__':
    ronlp = spacy.load('ro_core_news_lg')
    text = 'În ultimele 24 de ore au fost raportate 79 de decese, dintre care 40 s-au produs anterior intervalului de referință și au fost consemnate de INSP astăzi.'
    doc = ronlp(text)
    for sentence_head in get_sentences(doc):
        print(list(sentence_head.subtree))