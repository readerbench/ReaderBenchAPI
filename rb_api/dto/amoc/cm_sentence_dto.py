from rb_api.dto.two_mode_graph.two_mode_graph_dto import TwoModeGraphDTO
from rb_api.json_serialize import JsonSerialize

import json


class CMSentenceDTO(JsonSerialize):

    def __init__(self, text: str, index: int, graph: TwoModeGraphDTO):
        self.text = text
        self.index = index
        self.graph = graph

    
    def serialize(self):
        return json.dumps(self.__dict__)
