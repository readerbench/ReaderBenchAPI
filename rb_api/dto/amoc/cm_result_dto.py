from rb_api.dto.amoc.cm_sentence_dto import CMSentenceDTO
from rb_api.dto.amoc.cm_word_result_dto import CMWordResultDTO
from rb_api.json_serialize import JsonSerialize
import json

from typing import List
Sentences = List[CMSentenceDTO]
Words = List[CMWordResultDTO]


class CMResultDTO(JsonSerialize):

    def __init__(self, sentence_list: Sentences, word_list: Sentences):
        self.sentenceList = sentence_list
        self.wordList = word_list

    
    def serialize(self):
        return json.dumps(self.__dict__)
