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

def build_features(text: str, lang: Lang) -> Dict[str, float]:
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
    del cna_graph
    del doc
    if WordNet._instance is not None:
        del WordNet._instance.wn
    WordNet._instance = None
    pyphen.hdcache = {}
    gc.collect()
    return result
    