# Subiectul

"""
    Subiectul poate fi:
        - simplu
        - multiplu

    Partea de vorbire:
        - Substantiv
        - Pronume
    
    Informații:
        - număr
        - persoană
        - gen

    Atribute:
        - număr

    Subiect multiplu:
        - pentru fiecare informatiile de mai sus
"""

import spacy

class Subiect:

    class Componenta:

        def __init__(self, word) -> None:
            self.word = word
            self.type = word.tag_[0]

        def numar(self):
            if self.type == 'N':
                return self.word.tag_[3]
            elif self.type == 'P':
                return self.word.tag_[4]
            else:
                return None

        def persoana(self):
            if self.type == 'N':
                return 3
            elif self.type == 'P':
                return int(self.word.tag_[2])
            else:
                return None

        def gen(self):
            if self.type == 'N':
                return self.word.tag_[2]
            elif self.type == 'P':
                return self.word.tag_[3]
            else:
                return None
        
        def __str__(self) -> str:
            return self.word.text

        def __repr__(self) -> str:
            return self.word.text

    class Atribut:
        
        def __init__(self, word) -> None:
            self.word = word
            self.type = word.tag_[0]

        def numar(self):
            if self.type == 'N':
                return self.word.tag_[3]
            elif self.type == 'A':
                if len(self.word.tag_) < 5 or self.word.tag_[4] == '-':
                    return '*'
                return self.word.tag_[4]
            else:
                return None

        def persoana(self):
            if self.type == 'N':
                return 3
            return '*'

        def gen(self):
            if self.type == 'N':
                return self.word.tag_[2]
            elif self.type == 'A':
                if len(self.word.tag_) < 4 or self.word.tag[3] == '-':
                    return '*'
                return self.word.tag_[3]
            else:
                return None

        def __str__(self) -> str:
            return self.word.text

        def __repr__(self) -> str:
            return self.word.text


    def __init__(self, sent) -> None:
        self.type = None
        self.componente = []
        self.dictionar_componenta_atribute = {}
        self.coordination =  None
        self.diateza_pasiva = False
        self.subinteles = False
        self.populeaza_subiect(sent)

    def populeaza_subiect(self, sent):
        try:
            self.type, words, self.coordination, self.diateza_pasiva = get_subject_type(sent)
        except:
            self.subinteles = True
            return
        for word in words:
            componenta = self.Componenta(word)
            self.componente.append(componenta)
            attributes = get_attributes(word)
            self.dictionar_componenta_atribute[componenta] = [self.Atribut(atribut) for atribut in attributes]

    def __str__(self) -> str:
        return " ".join([str(x) for x in self.componente])
    
    def __repr__(self) -> str:
        return " ".join([str(x) for x in self.componente])


def get_subject_type(sent):
    for word in sent:
        if word.dep_.startswith('nsubj'):
            if word.children:
                other_subjects = []
                coord_type = None
                for child in word.children:
                    if child.dep_ == 'conj':
                        other_subjects.append(child)
                        for cc in child.children:
                            if cc.dep_ == 'cc':
                                if cc.tag_ == 'Ccssp':
                                    coord_type = 'disjunctiva'
                                elif cc.tag_ == 'Crssp':
                                    coord_type = 'conjunctiva'
                if other_subjects:
                    return 'multiplu', [word] + other_subjects, coord_type, word.dep_.endswith('pass')
            return 'simplu', [word], None, word.dep_.endswith('pass')


def get_attributes(word):
    return [child for child in word.children if child.dep_ in ["amod", "nmod", "nummod", "nmod:agent", "nmod:pmod", "nmod:tmod"]]


if __name__ == "__main__":
    nlp = spacy.load("ro_core_news_lg")
    text = "Băiatul frumos și cățelul vesel alergau prin parc"
    doc = nlp(text)
    subiect = Subiect(doc)
    print(subiect.componente[0].numar(), subiect.componente[0].gen())
    # for word in doc:
    #     print(word, word.dep_, word.pos_, list(word.children))
    # print(get_subject_type(doc))
    # text = "El și ea mergeau pe stradă"
    # doc = nlp(text)
    # print(get_subject_type(doc))
    # text = "Ionuț și Maria și George merg în parc"
    # doc = nlp(text)
    # print(get_subject_type(doc))
    # text = "Ea este acasă"
    # doc = nlp(text)
    # print(get_subject_type(doc))