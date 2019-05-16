import json
from rb_api.response import Response

class TextualComplexityResponse(Response):

    def __init__(self, data, errorMsg, success):
        self.data = data
        Response.__init__(self, data, errorMsg, success)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)

    def __str__(self):
        return "TextualComplexityResponse (success=%s, errorMsg='%s')\n%s\n" % (self.success, self.errorMsg, self.data)
