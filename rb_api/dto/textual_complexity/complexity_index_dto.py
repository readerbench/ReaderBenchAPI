import json
from rb_api.json_serialize import JsonSerialize

class ComplexityIndexDTO(JsonSerialize):

    def __init__(self, index, value, type, paragraph_index = -1, sentence_index=-1):
        self.index = index
        self.value = value
        self.type = type
        self.paragraph_index = paragraph_index
        self.sentence_index = sentence_index

    def __str__(self):
        return "ComplexityIndexDTO: index=%s ; value=%s\n" % (self.index, self.value)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
