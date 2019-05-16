from .. import Response

class KeywordsResponse(Response):

    def __init__(self, data, errorMsg, success):
        self.data = data
        Response.__init__(self, data, errorMsg, success)
