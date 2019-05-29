import json
from rb_api.json_serialize import JsonSerialize

class ScoreDTO(JsonSerialize):

    def __init__(self, model, score):
        self.model = str(model)
        self.score = str(score)

    def __str__(self):
        return "ScoreDTO: model=%s ; score=%s\n" % (self.model, self.score)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
