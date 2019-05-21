from rb_api.json_serialize import JsonSerialize

import json


class ArticleKeywordDTO(JsonSerialize):

    def __init__(self, value: str, type_: str):
        self.value = value
        self.type = type_
        self.scoreList = []


    def serialize(self):
        return json.dumps(self.__dict__)
