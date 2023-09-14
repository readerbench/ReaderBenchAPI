import gc
import logging
from typing import Dict

import pyphen
import torch
import tensorflow as tf
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.similarity.vector_model_factory import (VectorModelType,
                                                create_vector_model)
from rb.similarity.wordnet import WordNet
from rb.utils.rblogger import Logger

def build_features(text: str, lang: Lang, all_elements=False) -> Dict[str, float]:
    Logger.get_logger().setLevel(logging.WARNING)
    tf.config.run_functions_eagerly(True)
    tf.config.set_visible_devices([], 'GPU')
    doc = Document(lang, text)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, "")
    model.encode(doc)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph) 
    result = {
        str(index): float(value) if value is not None else None
        for index, value in doc.indices.items()
    }
    if all_elements:
        elements = [{"id": "Doc", "text": doc.text}]
        indices = [result]
        for i, block in enumerate(doc.get_blocks()):
            elements.append({"id": f"Paragraph_{i+1}", "text": block.text})
            indices.append({
                str(index): float(value) if value is not None else None
                for index, value in block.indices.items()
            })
            for j, sentence in enumerate(block.get_sentences()):
                elements.append({"id": f"Sentence_{i+1}.{j+1}", "text": sentence.text})
                indices.append({
                    str(index): float(value) if value is not None else None
                    for index, value in sentence.indices.items()
                })
        result = {
            "elements": elements,
            "indices": indices,
        }
    del cna_graph
    del doc
    if WordNet._instance is not None:
        del WordNet._instance.wn
    WordNet._instance = None
    pyphen.hdcache = {}
    gc.collect()
    return result
    