import json
from rb_api.json_serialize import JsonSerialize

class ComplexityIndexDTO(JsonSerialize):

    def __init__(self, index, value, key = "0", type = ""):
        self.index = index
        self.value = value
        self.key = key
        self.type = type

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
