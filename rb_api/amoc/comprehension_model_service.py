from rb_api.dto.amoc.cm_result_dto import CMResultDTO
from rb_api.dto.amoc.cm_sentence_dto import CMSentenceDTO
from rb_api.dto.two_mode_graph.two_mode_graph_dto import TwoModeGraphDTO
from rb_api.dto.two_mode_graph.two_mode_graph_node_dto import TwoModeGraphNodeDTO, TwoModeGraphNodeTypeDTO
from rb_api.dto.two_mode_graph.two_mode_graph_edge_dto import TwoModeGraphEdgeDTO
from rb_api.dto.amoc.cm_word_result_dto import CMWordResultDTO
from rb_api.dto.amoc.cm_word_activation_result_dto import CMWordActivationResultDTO

from rb.similarity.vector_model import VectorModel
from rb.comprehension.comprehension_model import ComprehensionModel
from rb.comprehension.utils.graph.cm_node_type import CmNodeType
from rb.comprehension.utils.graph.cm_edge_type import CmEdgeType
from rb.comprehension.utils.graph.cm_node_do import CmNodeDO
from rb.comprehension.utils.memory.word_activation import WordActivation
from rb.core.word import Word
from rb.core.lang import Lang
from typing import List, Dict
SemanticModels = List[VectorModel]
ListOfActivations = List[Dict[Word, WordActivation]]


class ComprehensionModelService():

    def __init__(self, semantic_models: SemanticModels, lang: Lang, min_activation_threshold: float,
                        max_active_concepts: float, max_semantic_expand: float):
        self.semantic_models = semantic_models
        self.lang = lang
        self.min_activation_threshold = min_activation_threshold
        self.max_active_concepts = max_active_concepts
        self.max_semantic_expand = max_semantic_expand


    def run(self, text: str) -> CMResultDTO:
        result = CMResultDTO([], [])
        
        cm = ComprehensionModel(text, self.lang, self.semantic_models,
                    self.min_activation_threshold, self.max_active_concepts, self.max_semantic_expand)

        for index in range(cm.get_total_number_of_phrases()):
            sentence = cm.get_sentence_at_index(index)

            # syntactic_indexer = cm.get_syntactic_indexer_at_index(index)
            # current_syntactic_graph = syntactic_indexer.get_cm_graph(CmNodeType.TextBased)
            # current_graph = cm.current_graph
            current_syntactic_graph = cm.sentence_graphs[index]
            current_graph = cm.current_graph

            current_graph.combine_with_syntactic_links(current_syntactic_graph, sentence, cm.semantic_models, cm.max_dictionary_expansion)

            cm.current_graph = current_graph
            cm.apply_page_rank(index)

            tmg = TwoModeGraphDTO()
            cm_sentence = CMSentenceDTO(sentence.text, index, tmg)

            for node in current_graph.node_list:
                text = node.word.lemma
                if node.node_type == CmNodeType.TextBased:
                    tmg_node = TwoModeGraphNodeDTO(TwoModeGraphNodeTypeDTO.TEXT_BASED, text, text)
                else:
                    tmg_node = TwoModeGraphNodeDTO(TwoModeGraphNodeTypeDTO.INFERRED, text, text)
                tmg_node.active = node.active
                tmg.nodeList.append(tmg_node)

            for edge in current_graph.edge_list:
                tmg_edge = TwoModeGraphEdgeDTO(edge.score, edge.node1.word.lemma, edge.node2.word.lemma, str(edge.edge_type.value) + 'Distance')
                tmg.edgeList.append(tmg_edge)
            
            cm.save_scores(cm.sentence_graphs[index])

            result.sentenceList.append(cm_sentence)

        history_keeper = cm.history_keeper
        for node in history_keeper.unique_word_list:
            if node.node_type == CmNodeType.TextBased:
                result.wordList.append(self.get_cm_word_result(node, history_keeper.activation_history))
            
        for node in history_keeper.unique_word_list:
            if node.node_type != CmNodeType.TextBased:
                result.wordList.append(self.get_cm_word_result(node, history_keeper.activation_history))
        
        return result


    def get_cm_word_result(self, node: CmNodeDO, activation_history: ListOfActivations):
        result = CMWordResultDTO(node.word.lemma, node.node_type, [])

        for activation_map in activation_history:
            if node.word not in list(activation_map.keys()):
                act_result = CMWordActivationResultDTO(0.0, False)
                result.activationList.append(act_result)
            else:
                word_activation = activation_map[node.word]
                act_result = CMWordActivationResultDTO(word_activation.activation_value, word_activation.active)
                result.activationList.append(act_result)
        
        return result
