# Predicatul

"""
    Predicatul:
    - predicat verbal
    - predicat nominal (verb copulativ + nume predicativ)

    verb:
        - mod
        - număr
        - persoană
    *modul influenteaza restul
"""

import spacy

class Predicat:

    def __init__(self, sent) -> None:
        self.type, self.verb, self.marks, self.auxs, self.nume_predicative, self.diateza_pasiva = get_predicate(sent)

    def persoana_verb(self):
        for aux in self.auxs:
            if aux.tag_[0] == 'V':
                if len(aux.tag_) < 5 or aux.tag_[4] == '-':
                    return '*'
                else:
                    return int(aux.tag_[4])
        if len(self.verb.tag_) < 5 or self.verb.tag_[4] == '-':
            return '*'
        else:
            return int(self.verb.tag_[4])

    def numar_verb(self):
        for aux in self.auxs:
            if aux.tag_[0] == 'V':
                if len(aux.tag_) < 6 or aux.tag_[5] == '-':
                    return '*'
                else:
                    return aux.tag_[5]
        if len(self.verb.tag_) < 6 or self.verb.tag_[5] == '-':
            return '*'
        else:
            return self.verb.tag_[5]

    def gen_nps(self):
        if not self.nume_predicative:
            return None
        results = []
        for np in self.nume_predicative:
            if np.tag_[0] == 'N':
                results.append(np.tag_[2])
            elif len(np.tag_) < 4 or np.tag_[3] == '-':
                results.append('*')
            else:
                results.append(np.tag_[3])
        return results

    def numar_nps(self):
        if not self.nume_predicative:
            return None
        results = []
        for np in self.nume_predicative:
            if np.tag_[0] == 'N':
                results.append(np.tag_[3])
            else:
                if len(np.tag_) < 5 or np.tag_[4] == '-':
                    results.append('*')
                else:
                    results.append(np.tag_[4])
        return results

    def caz_np(self):
        pass

    def __str__(self) -> str:
        return f'{str(self.auxs) + " " if self.auxs else ""}{str(self.marks) + " " if self.marks else ""}{self.verb}{" " + str(self.nume_predicative) if self.nume_predicative else ""}'


    def __repr__(self) -> str:
        return f'{str(self.auxs) + " " if self.auxs else ""}{str(self.marks) + " " if self.marks else ""}{self.verb}{" " + str(self.nume_predicative) if self.nume_predicative else ""}'


def get_predicate(sent):
    for word in sent:
        if word.dep_ == "ROOT":
            if word.tag_[0] == 'V': # verb
                marks = [child for child in word.children if child.dep_ == 'mark']
                auxs = [child for child in word.children if child.dep_ == 'aux']
                return 'verbal', word, marks, auxs, None, False
            elif word.tag_[0] in ['A', 'R', 'N']: # nume predicativ adj substantiv
                verb = None
                diateza_pasiva = None
                for child in word.children:
                    if child.dep_ == 'cop':
                        verb = child
                        diateza_pasiva = False
                        break
                    elif child.dep == 'aux:pass':
                        verb = child
                        diateza_pasiva = True
                        break
                marks = [child for child in verb.children if child.dep_ == 'mark']
                auxs = [child for child in word.children if child.dep_ == 'aux']
                if diateza_pasiva:
                    return 'verbal', verb, marks, auxs, [word] + [child for child in word.children if child.dep_ == 'conj'], diateza_pasiva
                else:
                    return 'nominal', verb, marks, auxs, [word] + [child for child in word.children if child.dep_ == 'conj'], diateza_pasiva
            else:
                print("ROOT-ul nu este nici verb, nici adjectiv, nici substantiv")
                return None, None, None, None, None


if __name__ == "__main__":
    nlp = spacy.load("ro_core_news_lg")
    text = "Nici băiatul, nici fata nu au plecat."
    doc = nlp(text)
    for word in doc:
        print(word, word.dep_, word.pos_, word.tag_, list(word.children))
    predicat = Predicat(doc)
    print(predicat.numar_verb())