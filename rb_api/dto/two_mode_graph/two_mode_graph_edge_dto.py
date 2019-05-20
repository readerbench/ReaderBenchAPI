

class TwoModeGraphEdgeDTO():

    def __init__(self, score: float, source_uri: str, target_uri: str):
        self.score = score
        self.source_uri = source_uri
        self.target_uri = target_uri


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