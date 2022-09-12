from re import sub
from services.subject_predicate.subiect import Subiect
from services.subject_predicate.predicat import Predicat
import spacy


def acord_s_simplu_pv(sent): # A 1, A 2, A 3
    s = Subiect(sent)
    p = Predicat(sent)
    if s.type != 'simplu':
        return True, None

    # if p.type != 'verbal':
    #     return True, None

    errors = []
    if p.numar_verb() != '*' and s.componente[0].numar() != p.numar_verb():
        errors.append('Subiectul și predicatul nu se acordă în număr')

    if p.persoana_verb() != '*' and s.componente[0].persoana() != p.persoana_verb():
        errors.append('Subiectul și predicatul nu se acordă în persoană')
    
    if errors:
        return False, errors
    return True, None


def acord_s_pv_diateza_pasiva_si_pn(sent, nlp): # A 1
    s = Subiect(sent)
    p = Predicat(sent)

    if p.diateza_pasiva == False and p.type != 'nominal':
        return True, None
    
    errors = []
    if p.numar_verb() != '*' and s.componente[0].numar() != p.numar_verb():
        errors.append('Subiectul și predicatul nu se acordă în număr')

    if p.persoana_verb() != '*' and s.componente[0].persoana() != p.persoana_verb():
        errors.append('Subiectul și predicatul nu se acordă în persoană')

    ok1, errors1 = acord_s_simplu_nume_predicativ_adjectiv(sent, True)
    ok2, errors2 = acord_s_multiplu_np(sent, nlp, True)

    errors = errors + errors1 if not ok1 else [] + errors2 if not ok2 else []
    if errors:
        return False, errors
    return True, None


def acord_s_simplu_plus_atr_pl_pv(sent): # A 5
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'simplu':
        return True, None

    # if p.type != 'verbal':
    #     return True, None

    if s.componente[0].word.lemma_ not in ['soi', 'specie', 'rasă', 'tip', 'fel']:
        return True, None

    ss = s.componente[0]
    if s.dictionar_componenta_atribute[ss] and p.numar_verb() != '*' and s.dictionar_componenta_atribute[ss][0].numar() != p.numar_verb():
        return False, ["Numărul atributului subiectului diferă de numărul verbului"]

    return True, [None]


def acord_s_multiplu_subs_pv(sent): # B 1
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None

    # if p.type != 'verbal':
    #     return True, None

    for componenta in s.componente: # toate sa fie substantive
        if componenta.type != 'N':
            return True, None
    
    if p.numar_verb() not in ['p', '*']:
        return False, ["Subiectul multiplu ce conține numai substantive se acordă la plural cu predicatul simplu."]

    return True, None


def acord_s_multiplu_coord_disjunc(sent): # B 3
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None

    # if p.type != 'verbal':
    #     return True, None

    if s.coordination == 'disjunctiva':
        all_single = True
        for componenta in s.componente:
            if componenta.numar() != 's':
                all_single = False
                break
        
        if all_single:
            if p.numar_verb() not in ['s', '*']:
                return False, ["Subiectul multiplu coordonat disjunctiv cu toate elementele la singular necesită predicat la singular."]
        else:
            if p.numar_verb() not in ['p', '*']:
                return False, ["Subiectul multiplu coordonat disjunctiv cu cel puțin un element la plural necesită predicat la plural."]

    return True, None


def acord_s_multiplu_cu_pron_pv(sent): # B 4
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None

    # if p.type != 'verbal':
    #     return True, None
    
    lowest_person = 10
    for componenta in s.componente:
        if componenta.type == 'P':
            if lowest_person > componenta.persoana():
                lowest_person = componenta.persoana()

    if lowest_person == 10: # nu am gasit niciun pronume
        return True, None
    
    errors = []
    if p.persoana_verb() != '*' and lowest_person != p.persoana_verb():
        errors.append("Predicatul verbal nu se acordă în persoană cu cea mai mică persoană a subiectul multiplu.")
    
    if p.numar_verb() not in ['p', '*']:
        errors.append("Predicatul nu este la plural.")
    
    if errors:
        return False, errors
    return True, None


def acord_subiect_predicat_non_verbal(sent):
    s = Subiect(sent)
    p = Predicat(sent)

    # if p.type == 'verbal':
    #     return True, None

    errors = []
    if s.type == 'multiplu':
        if p.numar_verb() not in ['*', 'p']:
            errors.append("Predicatul nu se acordă în număr cu subiectul multiplu.")
        


def acord_s_multiplu_pron_negative_cu_pv(sent): # B 5
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None

    if p.type != 'verbal':
        return True, None

    for componenta in s.componente:
        if componenta.type != 'P':
            return True, None
        if componenta.type == 'P' and componenta.word.tag_[1] != 'z':
            return True, None

    if p.numar_verb() != 's' and p.persoana_verb() not in ['*', 3]:
        return False, ['Subiectul multiplu format din pronume negative necesită predicat la persoana a 3-a singular.']
    
    return True, None


def acord_s_simplu_nume_predicativ_adjectiv(sent, bypass_p_type=False): # D 1
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'simplu':
        return True, None

    if p.type != 'nominal' and not bypass_p_type:
        return True, None

    ss = s.componente[0]
    errors = []
    for i, nr in enumerate(p.numar_nps()):
        if ss.numar() != nr:
            errors.append(f"Subiectul nu se acordă în număr cu numele predicativ {p.nume_predicative[i]}")
    for i, gen in enumerate(p.gen_nps()):
        if ss.gen() != gen:
            errors.append(f"Subiectul nu se acordă în gen cu numele predicativ {p.nume_predicative[i]}")
    
    if errors:
        return False, errors
    return True, None
    

def lucru_sau_fiinta(ref_lucru, ref_fiinta, word):
    if ref_lucru.similarity(word) > ref_fiinta.similarity(word):
        return 'lucru'
    else:
        return 'fiinta'


def doar_masculin_singular_si_altceva(subiect):
    altceva = False
    for componenta in subiect.componente:
        if componenta.gen() == 'm' and componenta.numar() =='p':
            return False
        if componenta.gen() != 'm':
            altceva = True
    return altceva


def doar_masculin_plural_feminim_singular_plural_sau_neutru_plural(subiect):
    for componenta in subiect.componente:
        if componenta.gen() == 'm' and componenta.numar() == 's':
            return False
        if componenta.gen() == 'n' and componenta.numar() == 's':
            return False
    return True


def doar_masculin_plural_neutru_singular(subiect):
    for componenta in subiect.componente:
        if componenta.gen() == 'm' and componenta.numar() == 's':
            return False
        if componenta.gen() == 'n' and componenta.numar() == 'p':
            return False
    return True


def acord_s_multiplu_np(sent, nlp, bypass_p_type=False): # D 2
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None
    
    if p.type != 'nominal' and not bypass_p_type:
        return True, None

    errors = []
    for i, nr in enumerate(p.numar_nps()):
        if 'p' != nr:
            errors.append(f"Subiectul multiplu nu se acordă în număr cu numele predicativ {p.nume_predicative[i]} care trebuie să fie la plural.")
    
    ref_lucru = nlp("lucru")[0]
    ref_fiinta = nlp("ființă")[0]

    genuri = [componenta.gen() for componenta in s.componente]
    tipuri = [lucru_sau_fiinta(ref_lucru, ref_fiinta, componenta.word) for componenta in s.componente]

    # D 2 1
    toate_fiinte = 'lucru' not in tipuri
    un_masculin = 'm' in genuri
    if toate_fiinte and un_masculin: 
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if 'm' != gen:
                local_errors.append(f"Subiectul multiplu nume de ființe cu cel puțin un masculin se acordă în genul masculin cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None

    # D 2 1
    if toate_fiinte and not un_masculin:
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if 'm' == gen:
                local_errors.append(f"Subiectul multiplu nume de ființe la genul feminin/neutru se acordă în genul feminin/neutru cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None

    # D 2 2
    unele_sunt_fiinte = 'fiinta' in tipuri
    if unele_sunt_fiinte:
        genurile_fiintelor = [genuri[i] for i, tip in enumerate(tipuri) if tip == 'fiinta']
        un_masculin = 'm' in genurile_fiintelor
        if un_masculin:
            local_errors = []
            for i, gen in enumerate(p.gen_nps()):
                if 'm' != gen:
                    local_errors.append(f"Subiectul multiplu ce conține nume de ființe cu cel puțin un masculin se acordă în genul masculin cu numele predicativ {p.nume_predicative[i]}")
            if local_errors:
                return False, errors + local_errors
            elif errors:
                return False, errors    
            return True, None
        else:
            local_errors = []
            for i, gen in enumerate(p.gen_nps()):
                if 'm' == gen:
                    local_errors.append(f"Subiectul multiplu ce conține nume de ființe la genul feminin/neutru se acordă în genul feminin/neutru cu numele predicativ {p.nume_predicative[i]}")
            if local_errors:
                return False, errors + local_errors
            elif errors:
                return False, errors    
            return True, None

    # D 2 3
    toate_lucruri = 'fiinta' not in tipuri
    niciun_masculin = 'm' not in genuri
    if toate_lucruri and niciun_masculin:
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if 'm' == gen:
                local_errors.append(f"Subiectul multiplu nume de lucruri cu genuri feminim/neutru se acordă în genul feminin/neutru cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None

    # D 2 4 + D 2 5
    doar_m_sg_si_altceva = doar_masculin_singular_si_altceva(s)
    if toate_lucruri and doar_m_sg_si_altceva:
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if 'm' == gen:
                local_errors.append(f"Subiectul multiplu nume de lucruri cu gen masculin nr singular si genuri feminim/neutru numar singular/plural se acordă în genul feminin/neutru cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None
    
    # D 2 6
    doar_m_pl_f_sg_pl_n_pl = doar_masculin_plural_feminim_singular_plural_sau_neutru_plural(s)
    if toate_lucruri and doar_m_pl_f_sg_pl_n_pl:
        last_gen = genuri[-1]
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if last_gen != gen:
                local_errors.append(f"Subiectul multiplu nume de lucruri cu gen masculin plural si gen feminim singular/plural si/sau gen neutru plural se acordă în genul celui mai apropiat termen cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None
    
    # D 2 7
    doar_m_pl_n_sg = doar_masculin_plural_neutru_singular(s)
    if toate_lucruri and doar_m_pl_n_sg:
        local_errors = []
        for i, gen in enumerate(p.gen_nps()):
            if 'm' != gen:
                local_errors.append(f"Subiectul multiplu nume de lucruri cu gen masculin nr plural si gen neutru nr singular se acordă în genul masculin cu numele predicativ {p.nume_predicative[i]}")
        if local_errors:
            return False, errors + local_errors
        elif errors:
            return False, errors    
        return True, None

    return True, None

def acord_s_nume_predicativ_substantiv(sent): # E
    s = Subiect(sent)
    p = Predicat(sent)

    if p.type != 'nominal':
        return True, None
    
    numar_subiect = None
    if s.type == 'simplu':
        numar_subiect = s.componente[0].numar()
    else:
        numar_subiect = 'p'

    errors = []
    for i, nr in enumerate(p.numar_nps()):
        if numar_subiect != nr:
            errors.append(f"Subiectul nu se acordă în număr cu numele predicativ {p.nume_predicative[i]}")
    
    if errors:
        return False, errors
    return True, None


def exceptie_s_simplu_atr_pl(sent): # A 4
    s = Subiect(sent)

    if s.type != 'simplu':
        return False, None
    
    if s.componente[0].word.lemma_ in ['mulțime', 'grămadă', 'majoritate', 'sumedenie'] and s.dictionar_componenta_atribute[s.componente[0]] and \
        s.dictionar_componenta_atribute[s.componente[0]].numar() == 'p':
        return True, None
    
    return False, None


def exceptie_s_simplu_cons_partitive(sent): # A 6
    s = Subiect(sent)

    if s.type != 'simplu':
        return False, None
    
    if s.componente[0].word.lemma_ in ['sfert', 'pereche', 'parte', 'jumătate', 'duzină'] and s.dictionar_componenta_atribute[s.componente[0]]:
        return True, None
    
    return False, None


def exceptie_s_multiplu_coord_neg(sent): # B 2 exceptie
    s = Subiect(sent)

    if s.type != 'multiplu':
        return False, None

    for componenta in s.componente:
        if componenta.numar() != 's':
            return False, None
        nici_here = False
        for x in componenta.word.children:
            if x.lemma_ in ['Nici', 'nici']:
                nici_here = True
        if not nici_here:
            return False, None
        
    return True, None


def s_multiplu_coord_neg_ultimul_pl(sent): # B 2 not exceptie
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'multiplu':
        return True, None

    if s.componente[-1].numar() != 'p':
        return True, None
    
    for componenta in s.componente:
        nici_here = False
        for x in componenta.word.children:
            if x.lemma_ in ['Nici', 'nici']:
                nici_here = True
        if not nici_here:
            return True, None

    if p.numar_verb() not in ['p', '*']:
        return False, ['Pentru subiectul multiplu coordonat negativ cu ultima componenta la plural, predicatul trebuie sa fie si el la plural']
    
    return True, None


def exceptie_s_pronume_politete(sent): # C
    s = Subiect(sent)
    p = Predicat(sent)

    if s.type != 'simplu':
        return False, None

    if s.componente[0].type != 'P':
        return False, None
    
    if s.componente[0].word.text.lower().startswith('dumnea') and p.numar_verb not in ['*', 'p']:
        return True, ['Verbul se acordă la plural cu pronumele de politețe.']

    return False, None


def text_check(sent, ro_model):
    subiect = Subiect(sent)
    if subiect.subinteles:
        return []

    # C
    flag, errors = exceptie_s_pronume_politete(sent)
    if flag:
        return errors
    
    # A 4
    flag, errors = exceptie_s_simplu_atr_pl(sent)
    if flag:
        return errors

    # A 6
    flag, errors = exceptie_s_simplu_cons_partitive(sent)
    if flag:
        return errors

    # A 5
    flag, errors = acord_s_simplu_plus_atr_pl_pv(sent)
    if not flag:
        return errors
    if type(errors) != list:    
        # A 1, A 2, A 3
        flag, errors = acord_s_simplu_pv(sent)
        if not flag:
            return errors
    
    # B 2 not exceptie
    flag, errors = s_multiplu_coord_neg_ultimul_pl(sent)
    if not flag:
        return errors
    
    # B 2 exceptie
    flag, errors = exceptie_s_multiplu_coord_neg(sent)
    if not flag:
        # B 3
        flag, errors = acord_s_multiplu_coord_disjunc(sent)
        if not flag:
            return errors
        
        # B 4
        flag, errors = acord_s_multiplu_cu_pron_pv(sent)
        if not flag:
            return errors
        
        # B 5
        flag, errors = acord_s_multiplu_pron_negative_cu_pv(sent)
        if not flag:
            return errors

        # B 1
        flag, errors = acord_s_multiplu_subs_pv(sent)
        if not flag:
            return errors

    # D 1
    flag, errors = acord_s_simplu_nume_predicativ_adjectiv(sent)
    if not flag:
        return errors

    # D 2
    flag, errors = acord_s_multiplu_np(sent, ro_model)
    if not flag:
        return errors

    # E
    flag, errors = acord_s_nume_predicativ_substantiv(sent)
    if not flag:
        return errors

    return []


def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT"):
            root_token = token
    return root_token

def find_other_verbs(doc, root_token):
    other_verbs = []
    for token in doc:
        ancestors = list(token.ancestors)
        if (token.pos_ == "VERB" and len(ancestors) == 1\
            and ancestors[0] == root_token):
            other_verbs.append(token)
    return other_verbs

def get_clause_token_span_for_verb(verb, doc, all_verbs):
    first_token_index = len(doc)
    last_token_index = 0
    this_verb_children = list(verb.children)
    for child in this_verb_children:
        if (child not in all_verbs):
            if (child.i < first_token_index):
                first_token_index = child.i
            if (child.i > last_token_index):
                last_token_index = child.i
    return(first_token_index, last_token_index)

def split_sentence_into_clauses(doc):
    root_token = find_root_of_sentence(doc)
    other_verbs = find_other_verbs(doc, root_token)
    token_spans = []   
    all_verbs = [root_token] + other_verbs
    for other_verb in all_verbs:
        (first_token_index, last_token_index) = \
        get_clause_token_span_for_verb(other_verb, 
                                        doc, all_verbs)
        token_spans.append((first_token_index, 
                            last_token_index))
    sentence_clauses = []
    for token_span in token_spans:
        start = token_span[0]
        end = token_span[1]
        if (start < end):
            clause = doc[start:end]
            sentence_clauses.append(clause)
    sentence_clauses = sorted(sentence_clauses, 
                            key=lambda tup: tup[0])

    clauses_text = [clause.text for clause in sentence_clauses]
    return clauses_text