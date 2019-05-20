from rb_api.dto.two_mode_graph.two_mode_graph_node_dto import TwoModeGraphNodeDTO, TwoModeGraphNodeTypeDTO

class TwoModeGraphDegreeNodeDTO(TwoModeGraphNodeDTO):

    def __init__(self, node_type: TwoModeGraphNodeTypeDTO, uri: str, display_name: str, degree: float):
        super().__init__(node_type, uri, display_name)
        self.degree = degree