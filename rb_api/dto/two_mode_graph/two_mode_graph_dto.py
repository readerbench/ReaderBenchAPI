from rb_api.dto.two_mode_graph.two_mode_graph_node_dto import TwoModeGraphNodeDTO
from rb_api.dto.two_mode_graph.two_mode_graph_edge_dto import TwoModeGraphEdgeDTO
from rb_api.json_serialize import JsonSerialize

import json


class TwoModeGraphDTO(JsonSerialize):
    
    def __init__(self):
        self.edgeList = []
        self.nodeList = []


    def add_edge(self, edge: TwoModeGraphEdgeDTO):
        self.edgeList.append(edge)


    def add_node(self, node: TwoModeGraphNodeDTO):
        self.nodeList.append(node) 


    def serialize(self):
        return json.dumps(self.__dict__)
