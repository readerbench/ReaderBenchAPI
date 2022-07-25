from typing import Dict
from django.http import JsonResponse
from django.shortcuts import render
from rb import Document, Lang
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.core.text_element import TextElement, TextElementType
from rb.similarity.transformers_encoder import TransformersEncoder
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.utils.utils import str_to_lang
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated


def build_text_element_result(elem: TextElement) -> Dict:
    if elem.depth < TextElementType.SENT.value:
        return {
            "text": elem.text
        }
    return {
        "text": elem.text,
        "indices": sorted([
                (str(index), value) 
                for index, value in elem.indices.items()
            ], 
            key=lambda x: x[0]),
        "children": [
            build_text_element_result(child) 
            for child in elem.components
        ]
    }


# Create your views here.
@api_view(['POST'])
@permission_classes([AllowAny])
def get_indices(request):
    text = request.data["text"]
    lang = request.data["lang"]
    lang = str_to_lang(lang)
    doc = Document(lang=lang, text=text)
    encoder: TransformersEncoder = create_vector_model(lang, VectorModelType.TRANSFORMER, None)
    encoder.encode(doc)
    graph = CnaGraph(doc, models=[encoder])
    compute_indices(doc, graph)
    response = build_text_element_result(doc)
    return JsonResponse(response, safe=False)
