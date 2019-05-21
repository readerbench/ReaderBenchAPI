from rb_api.json_serialize import JsonSerialize

import json


class TwoModeGraphEdgeDTO(JsonSerialize):

    def __init__(self, score: float, source_uri: str, target_uri: str, edge_type: str):
        self.score = score
        self.sourceUri = source_uri
        self.targetUri = target_uri
        self.edgeType = edge_type


    def __lt__(self, other):
        if isinstance(other, TwoModeGraphEdgeDTO):
            return self.score < other.score
        return NotImplemented


    def __gt__(self, other):
        if isinstance(other, TwoModeGraphEdgeDTO):
            return self.score > other.score
        return NotImplemented


    def __eq__(self, other):
        if isinstance(other, TwoModeGraphEdgeDTO):
            return self.score == other.score
        return NotImplemented


    def serialize(self):
        return json.dumps(self.__dict__)
