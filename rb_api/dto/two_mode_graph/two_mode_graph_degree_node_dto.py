from rb_api.dto.two_mode_graph.two_mode_graph_node_dto import TwoModeGraphNodeDTO, TwoModeGraphNodeTypeDTO
from rb_api.json_serialize import JsonSerialize

import json


class TwoModeGraphDegreeNodeDTO(TwoModeGraphNodeDTO, JsonSerialize):

    def __init__(self, node_type: TwoModeGraphNodeTypeDTO, uri: str, display_name: str, degree: float):
        super().__init__(node_type, uri, display_name)
        self.degree = degree

    
    def serialize(self):
        return json.dumps(self.__dict__)
