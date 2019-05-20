from rb_api.dto.amoc.cm_sentence_dto import CMSentenceDTO
from rb_api.dto.amoc.cm_word_result_dto import CMWordResultDTO
from typing import List
Sentences = List[CMSentenceDTO]
Words = List[CMWordResultDTO]


class CMResultDTO():

    def __init__(self, sentence_list: Sentences, word_list: Sentences):
        self.sentence_list = sentence_list
        self.word_list = word_list