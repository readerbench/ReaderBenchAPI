from rb_api.dto.two_mode_graph.two_mode_graph_node_type_dto import TwoModeGraphNodeTypeDTO
from rb_api.json_serialize import JsonSerialize

import json


class TwoModeGraphNodeDTO(JsonSerialize):
    
    def __init__(self, node_type: TwoModeGraphNodeTypeDTO, uri: str, display_name: str):
        self.type = node_type
        self.uri = uri
        self.displayName = display_name
        self.active = True
    
    def __repr__(self):
        return "TwoModeGraphNodeDTO(%r)" % self.uri


    def __str__(self):
        return "TwoModeGraphNodeDTO(%r)" % self.uri


    def serialize(self):
        return json.dumps(self.__dict__)
