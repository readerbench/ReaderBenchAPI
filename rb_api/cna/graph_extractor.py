from typing import Dict, List

from rb.cna.cna_graph import CnaGraph
from rb.cna.edge_type import EdgeType
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.text_element import TextElement
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.word2vec import Word2Vec


def encode_element(element: TextElement, names: Dict[TextElement, str], graph: CnaGraph):
    result =  { "name": names[element], "value": element.text, "type": element.depth, "importance": graph.importance[element] }
    if not element.is_sentence():
        result["children"] = [encode_element(child, names, graph) for child in element.components]
    return result

def compute_graph(texts: List[str], lang: Lang, models: List) -> str:
    docs = [Document(lang=lang, text=text) for text in texts]
    models = [create_vector_model(lang, VectorModelType.from_str(model["model"]), model["corpus"]) for model in models]
    models = [model for model in models if model is not None]
    graph = CnaGraph(docs=docs, models=models)
    sentence_index = 1
    doc_index = 1
    names = {}
    for doc_index, doc in enumerate(docs):
        names[doc] = "Document {}".format(doc_index+1)
        for paragraph_index, paragraph in enumerate(doc.components):
            names[paragraph] = "Paragraph {}.{}".format(doc_index+1, paragraph_index+1)
            for sentence_index, sentence in enumerate(paragraph.components):
                names[sentence] = "Sentence {}.{}".format(doc_index + 1, sentence_index + 1)
    result = {"data": {
        "name": "Document Set", "value": None, "type": None, "importance": None,
        "children": [encode_element(doc, names, graph) for doc in docs]}
        }
    edges = {}
    for a, b, data in graph.graph.edges(data=True):
        if data["type"] is not EdgeType.ADJACENT and data["type"] is not EdgeType.PART_OF:
            if data["type"] is EdgeType.COREF:
                edge_type = EdgeType.COREF.name
            else:
                edge_type = "{}: {}".format(data["type"].name, data["model"].name)
            if (names[a], names[b]) not in edges:
                edges[(names[a], names[b])] = []
            edge = {
                "name": edge_type,
                "weight": str(data["value"]) if "value" in data else None,
                "details": data["details"] if "details" in data else None,
            }
            edges[(names[a], names[b])].append(edge)
    edges = [
        {
            "source": pair[0],
            "target": pair[1],
            "types": types,
        }
        for pair, types in edges.items()
    ]
    result["data"]["edges"] = edges
    return result
