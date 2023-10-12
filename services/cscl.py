from typing import Dict

from rb.cna.cna_graph import CnaGraph
from rb.core.cscl.contribution import Contribution
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cscl_parser import load_from_xml
from rb.processings.cscl.participant_evaluation import (
    evaluate_interaction, evaluate_involvement, evaluate_textual_complexity,
    perform_sna)
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.utils import str_to_lang


def process_conv(file, lang):
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')
    conv_dict = load_from_xml(file)
    
    lang = str_to_lang(lang)
    model = create_vector_model(lang, VectorModelType.TRANSFORMER, None)
    conv = Conversation(lang, conv_dict, apply_heuristics=False)
    model.encode(conv)
    conv.graph = CnaGraph(conv, models=[model], pairwise=False, window=30)
    evaluate_interaction(conv)
    evaluate_involvement(conv)
    participant_graph = perform_sna(conv, False)

    contr_kb = {
        contribution: 0
        for contribution in conv.components
    }
    contr_in = {
        contribution: 0
        for contribution in conv.components
    }
    contr_out = {
        contribution: 0
        for contribution in conv.components
    }
    contr_edges = []
    for u, v, w in conv.graph.filtered_graph.edges.data('weight', default=None):
        if isinstance(u, Contribution) and isinstance(v, Contribution) and u.index < v.index:
            contr_in[u] += w
            contr_out[v] += w
            contr_edges.append({
                "source": u.index, 
                "target": v.index,
                "weight": w
            })
            if u.get_participant() != v.get_participant():
                contr_kb[u] += w
                contr_kb[v] += w

    result = {
        "graph": {
            "participants": [p.get_id() for p in participant_graph.nodes],
            "edges": [
                {
                    "source": a.get_id(), 
                    "target": b.get_id(),
                    "weight": w
                }
                for a, b, w in participant_graph.edges.data("weight")
            ]
        }, 
        "participants": {
            p.get_id(): {str(index): value for index, value in p.indices.items()}
            for p in conv.get_participants()
        },
        "contributions": [
            {
                "id": c.index,
                "text": c.text,
                "participant": c.get_participant().get_id(),
                "ref": c.parent_contribution.index if c.parent_contribution is not None else None,
                "time": c.timestamp.isoformat(),
                "importance": conv.graph.importance[c],
                "social_kb": contr_kb[c],
                "in_degree": contr_in[c],
                "out_degree": contr_out[c]
            }
            for c in conv.components
        ],
        "contribution_edges": contr_edges
    }
    return result