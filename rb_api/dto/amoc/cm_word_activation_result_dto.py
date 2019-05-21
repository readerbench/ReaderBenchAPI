from rb_api.json_serialize import JsonSerialize

import json


class CMWordActivationResultDTO():

    def __init__(self, score: float, active: bool):
        self.score = score
        self.isActive = active

    def serialize(self):
        return json.dumps(self.__dict__)
