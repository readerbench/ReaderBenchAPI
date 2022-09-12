from typing import Dict, List, Tuple

from rb.core.lang import Lang
from rb.core.document import Document
from rb.core.text_element_type import TextElementType
from rb.processings.keywords.keywords_extractor import extract_keywords
from rb.similarity.vector_model import VectorModel
from rb.similarity.vector_model_factory import get_default_model


def keywords_heatmap(text: str, lang: Lang = Lang.RO, granularity: TextElementType = TextElementType.SENT,
                         max_keywords: int = 40) -> Dict:
    vector_model: VectorModel = get_default_model(lang=lang)
    keywords: List[Tuple[float, str]] = extract_keywords(text=text, lang=lang, vector_model=vector_model)
    doc: Document = Document(lang=lang, text=text)

    elements, word_scores = {}, {}

    for kw in keywords:
        word_scores[kw[1]] = {}

    if granularity is TextElementType.SENT:
        for i, sent in enumerate(doc.get_sentences()):
            elements[str(i + 1)] = sent.text
            sent_vector = vector_model.get_vector(sent)
            for kw in keywords:
                word_scores[kw[1]][str(
                    i + 1)] = str(max(vector_model.similarity(vector_model.get_vector(kw[1]), sent_vector), 0))
    else:
        for i, block in enumerate(doc.get_blocks()):
            elements[str(i + 1)] = block.text
            block_vector = vector_model.get_vector(block)
            for kw in keywords:
                word_scores[kw[1]][str(
                    i + 1)] = str(max(vector_model.similarity(vector_model.get_vector(kw[1]), block_vector), 0))

    return {
            "elements": elements,
            "heatmap": {
                "wordScores":
                    word_scores
            }
    }


def transform_for_visualization(keywords: List[Tuple[int, str]], lang: Lang) -> Dict:
    vector_model: VectorModel = get_default_model(lang=lang)
    edge_list, node_list = [], []

    for i, kw1 in enumerate(keywords):
        kw1_vector = vector_model.get_vector(kw1[1])
        for j, kw2 in enumerate(keywords):
            kw2_vector = vector_model.get_vector(kw2[1])
            if i != j and vector_model.similarity(kw1_vector, kw2_vector) >= 0.3:
                edge_list.append({
                    "edgeType": "SemanticDistance",
                    "score": str(max(vector_model.similarity(kw1_vector, kw2_vector), 0)),
                    "sourceUri": kw1[1],
                    "targetUri": kw2[1]
                })

    for kw in keywords:
        node_list.append({
            "type": "Word",
            "uri": kw[1],
            "displayName": kw[1],
            "active": True,
            "degree": str(max(0, float(kw[0])))
        })

    return {
            "edgeList": edge_list,
            "nodeList": node_list
        }
