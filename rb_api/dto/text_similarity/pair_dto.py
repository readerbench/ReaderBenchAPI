import json
from rb_api.json_serialize import JsonSerialize

class PairDTO(JsonSerialize):

    def __init__(self, idText1, idText2, scores):
        self.idText1 = idText1
        self.idText2 = idText2
        self.scores = scores

    def __str__(self):
        return "PairDTO: idText1=%s ; idText2=%s\nscores=%s\n" % (self.idText1, self.idText2, self.scores)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
