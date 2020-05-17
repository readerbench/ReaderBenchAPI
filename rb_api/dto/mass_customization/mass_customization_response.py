import json
from rb_api.response import Response
from rb_api.json_serialize import JsonSerialize

class MassCustomizationResponse(Response, JsonSerialize):

    def __init__(self, data, errorMsg, success):
        self.data = data
        Response.__init__(self, data, errorMsg, success)

    def __str__(self):
        return "(success=%s, errorMsg='%s', data='%s')" % (self.success, self.errorMsg, self.data)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
