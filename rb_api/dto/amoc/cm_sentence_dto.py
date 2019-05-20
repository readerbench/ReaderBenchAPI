from rb_api.dto.two_mode_graph.two_mode_graph_dto import TwoModeGraphDTO

class CMSentenceDTO():

    def __init__(self, text: str, index: int, graph: TwoModeGraphDTO):
        self.text = text
        self.index = index
        self.graph = graph