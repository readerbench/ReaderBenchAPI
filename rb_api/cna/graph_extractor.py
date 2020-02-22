from typing import Dict, List

from rb.cna.cna_graph import CnaGraph
from rb.cna.edge_type import EdgeType
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.word2vec import Word2Vec


def encode_element(element: TextElement, names: Dict[TextElement, str]):
    result =  { "name": names[element], "value": element.text, "type": element.depth }
    if not element.is_sentence():
        result["children"] = [encode_element(child, names) for child in element.components]
    return result

def compute_graph(texts: List[str], lang: Lang, models: List) -> str:
    docs = [Document(lang=lang, text=text) for text in texts]
    models = [create_vector_model(lang, VectorModelType.from_str(model["model"]), model["corpus"]) for model in models]
    models = [model for model in models if model is not None]
    graph = CnaGraph(docs=docs, models=models)
    paragraph_index = 1
    sentence_index = 1
    names = {}
    for node in graph.graph.nodes():
        if node.is_document():
            names[node] = "Document"
        elif node.is_block():
            names[node] = "Paragraph {}".format(paragraph_index)
            paragraph_index += 1
        elif node.is_sentence():
            names[node] = "Sentence {}".format(sentence_index)
            sentence_index += 1
    result = {"data": {
        "name": None, "value": None, "type": None, 
        "children": [encode_element(doc, names) for doc in docs]}
        }
    edges = {}
    for a, b, data in graph.graph.edges(data=True):
        if data["type"] is not EdgeType.ADJACENT and data["type"] is not EdgeType.PART_OF:
            edge_type = "{}: {}".format(data["type"].name, data["model"].name)
            if edge_type not in edges:
                edges[edge_type] = []
            edges[edge_type].append({
                "source": names[a],
                "target": names[b],
                "weight": str(data["value"]),
            })
    result["data"]["edges"] = edges
    return result