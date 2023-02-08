from typing import Dict
from rb import Document, Lang
from rb.similarity.vector_model_factory import create_vector_model, VectorModelType
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices

def build_features(text: str, lang: Lang) -> Dict[str, float]:
    doc = Document(lang, text)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, "")
    model.encode(doc)
    cna_graph = CnaGraph(docs=doc, models=[model])
    compute_indices(doc=doc, cna_graph=cna_graph) 
    return {
        str(index): value
        for index, value in doc.indices.items()
    }