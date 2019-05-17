import json
from rb_api.json_serialize import JsonSerialize
class ComplexityIndicesDTO(JsonSerialize):

    def __init__(self, category, valences):
        self.category = category
        self.valences = valences

    def __str__(self):
        return "ComplexityIndicesDTO: category=%s ; valences=%s\n" % (self.category, self.valences)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
