import json
from rb_api.json_serialize import JsonSerialize
class Response(JsonSerialize):

    def __init__(self, data, errorMsg, success):
        self.data = data
        self.errorMsg = errorMsg
        self.success = success

    def __str__(self):
        return "Response (success=%s, errorMsg='%s')\n%s\n" % (self.success, self.errorMsg, self.data)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
