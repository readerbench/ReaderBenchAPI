import json
from rb_api.json_serialize import JsonSerialize

class CsclDataDTO(JsonSerialize):

    def __init__(self, lang, conceptMaps, csclIndices, csclIndicesDescriptions, participantEvolution, participantInteractionGraph, socialKB, contributionsIndices):
        self.lang = lang
        self.conceptMaps = conceptMaps
        self.csclIndices = csclIndices
        self.csclIndicesDescriptions = csclIndicesDescriptions
        self.participantEvolution = participantEvolution
        self.participantInteractionGraph = participantInteractionGraph
        self.socialKB = socialKB
        self.contributionsIndices = contributionsIndices

    def __str__(self):
        return "CsclData (lang=%s):\Concept Maps=%s\nCSCL Indices=%s\nCSCL Indices Descriptions=%s\nParticipant Evolution=%s\nParticipant Interaction Graph=%s\nSocial KB=%s" % (self.lang, self.conceptMaps, 
            self.csclIndices, self.csclIndicesDescriptions, self.participantEvolution, self.participantInteractionGraph, self.socialKB)

    def serialize(self):
        return json.dumps(self.__dict__)

    def __repr__(self):
        return self.serialize()

    @staticmethod
    def dumper(obj):
        if "serialize" in dir(obj):
            return obj.serialize()

        return obj.__dict__
