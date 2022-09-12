from rb.core.lang import Lang
from rb.core.word import Word


def find_mistakes_intervals(text, restored_text):
    mistakes_intervals = []
    i = 0
    while i < len(text):
        if text[i] != restored_text[i]:
            # go left
            j = i
            while j >= 0 and text[j].isalpha():
                j -= 1
            j += 1

            # go right
            k = i
            while k < len(text) and text[k].isalpha():
                k += 1
            k -= 1

            if check_word_in_dict(restored_text[j:(k+1)], Lang.RO):
                mistakes_intervals.append([j, k, text[j:(k+1)], restored_text[j:(k+1)]])

            i = k + 1
        else:
            i += 1

    return mistakes_intervals


def check_word_in_dict(word, lang):
    spacy_word = Word.from_str(lang, word)
    return spacy_word.is_dict_word()