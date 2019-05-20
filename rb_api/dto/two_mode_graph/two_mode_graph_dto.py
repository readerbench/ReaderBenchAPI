from rb_api.dto.two_mode_graph.two_mode_graph_node_dto import TwoModeGraphNodeDTO
from rb_api.dto.two_mode_graph.two_mode_graph_edge_dto import TwoModeGraphEdgeDTO


class TwoModeGraphDTO():
    
    def __init__(self):
        self.edge_list = []
        self.node_list = []


    def add_edge(self, edge: TwoModeGraphEdgeDTO):
        self.edge_list.append(edge)


    def add_node(self, node: TwoModeGraphNodeDTO):
        self.node_list.append(node) 
