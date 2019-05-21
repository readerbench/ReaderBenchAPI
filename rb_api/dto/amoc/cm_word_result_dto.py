from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb_api.dto.amoc.cm_word_activation_result_dto import CMWordActivationResultDTO
from rb_api.json_serialize import JsonSerialize

import json

from typing import List
Activations = List[CMWordActivationResultDTO]

class CMWordResultDTO(JsonSerialize):

    def __init__(self, value: str, node_type: CmNodeType, activation_list: Activations):
        self.value = value
        self.type = node_type
        self.activationList = activation_list


    def serialize(self):
        return json.dumps(self.__dict__)
