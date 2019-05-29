import json
from rb_api.json_serialize import JsonSerialize

class ScoresDTO(JsonSerialize):

    def __init__(self, language, corpus, pairs):
        self.language = language
        self.corpus = corpus
        self.pairs = pairs

    def __str__(self):
        return "ScoresDTO: \n%s\n" % (self.pairs)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
