from rb_api.dto.two_mode_graph.two_mode_graph_node_type_dto import TwoModeGraphNodeTypeDTO

class TwoModeGraphNodeDTO():
    
    def __init__(self, node_type: TwoModeGraphNodeTypeDTO, uri: str, display_name: str):
        self.type = node_type
        self.uri = uri
        self.display_name = display_name
        self.active = True
    
    def __repr__(self):
        return "TwoModeGraphNodeDTO(%r)" % self.uri


    def __str__(self):
        return "TwoModeGraphNodeDTO(%r)" % self.uri
