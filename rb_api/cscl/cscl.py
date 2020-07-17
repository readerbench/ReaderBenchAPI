import json
import os
from os import path
from time import time

from flask import Flask, Response, jsonify, request
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.complexity.index_category import IndexCategory
from rb.core.cscl.cna_indices_enum import CNAIndices
from rb.core.cscl.community import Community
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cscl_indices_descriptions import CsclIndicesDescriptions
from rb.core.cscl.cscl_parser import load_from_xml
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.processings.cscl.participant_evaluation import (
    evaluate_interaction, evaluate_involvement, evaluate_textual_complexity,
    perform_sna)
from rb.processings.keywords.keywords_extractor import KeywordExtractor
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import (VECTOR_MODELS,
                                                create_vector_model)
from rb.utils.utils import str_to_lang
from werkzeug.utils import secure_filename

from rb_api.dto.cscl.cscl_data_dto import CsclDataDTO
from rb_api.dto.cscl.cscl_response import CsclResponse

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['xml'])
UPLOAD_FOLDER = "upload"

def csclOption():
    return ""

def csclPost():
    params = json.loads(request.get_data())
    csclFile = params.get('cscl-file')
    languageString = params.get('language')
    lang = str_to_lang(languageString)

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "..", "upload", csclFile))
    conv_thread = load_from_xml(lang, filepath)
    conv = Conversation(lang=lang, conversation_thread=conv_thread, apply_heuristics=False)
    fr_le_monde_word2vec = create_vector_model(lang, VectorModelType.from_str("word2vec"), "le_monde_small")
    conv.graph = CnaGraph(docs=[conv], models=[fr_le_monde_word2vec])

    participant_list = conv.get_participants()  
    names = [p.get_id() for p in participant_list]
    
    conceptMap = {
        "nodeList": [],
        "edgeList": [],
    }

    # Begin Concept Map
    keywords_extractor = KeywordExtractor()
    keywords = keywords_extractor.extract_keywords(text=conv.text, lang=lang, vector_model=fr_le_monde_word2vec)
    for score, word in keywords:
        posStr = word.pos.value
        conceptMap["nodeList"].append(
            {
                "type": "Word",
                "uri": word.lemma + '_' + posStr,
                "displayName": word.lemma,
                "active": True,
                "degree": score
            }
        )
    for _, p in keywords:
        for _, q in keywords:
            posWord1 = p.pos.value
            posWord2 = q.pos.value
            conceptMap["edgeList"].append(
                {
                    "edgeType": "SemanticDistance",
                    "score": fr_le_monde_word2vec.similarity(p, q),
                    "sourceUri": p.lemma + '_' + posWord1,
                    "targetUri": q.lemma + '_' + posWord2
                }
            )
    # End Concept Map

    evaluate_interaction(conv)
    evaluate_involvement(conv)
    perform_sna(conv, False)
    evaluate_textual_complexity(conv)
    
    # Begin Participant Interaction Graph
    participantInteractionGraph = {
        "nodeList": [],
        "edgeList": [],
    }
    nameIndex = {}
    for i, n in enumerate(names):
        participantInteractionGraph["nodeList"].append(
            {
                "type": "Author",
                "uri": i,
                "displayName": n,
                "active": True
            },
        )
        nameIndex[n] = i
        
    for n1 in names:
        for n2 in names:
            participantInteractionGraph["edgeList"].append(
                {
                    "edgeType": "SemanticDistance",
                    "score": conv.get_score(n1, n2),
                    "sourceUri": nameIndex[n1],
                    "targetUri": nameIndex[n2]
                },
            )
    # End Participant Interaction Graph

    # Begin CSCL Indices
    csclIndices = {}

    for p in participant_list:
        participantDict = {
            "SCORE": p.get_index(CNAIndices.SCORE),
            "SOCIAL_KB": p.get_index(CNAIndices.SOCIAL_KB),
            "OUTDEGREE": p.get_index(CNAIndices.OUTDEGREE),
            "INDEGREE": p.get_index(CNAIndices.INDEGREE),
            "NO_NEW_THREADS": p.get_index(CNAIndices.NO_NEW_THREADS),
            "NEW_THREADS_OVERALL_SCORE": p.get_index(CNAIndices.NEW_THREADS_OVERALL_SCORE),
            "NEW_THREADS_CUMULATIVE_SOCIAL_KB": p.get_index(CNAIndices.NEW_THREADS_CUMULATIVE_SOCIAL_KB),
            "AVERAGE_LENGTH_NEW_THREADS": p.get_index(CNAIndices.AVERAGE_LENGTH_NEW_THREADS)
        }
        csclIndices[p.get_id()] = participantDict
    # End CSCL Indices

    # Begin CSCL Descriptions
    csclIndicesDescriptions = {}
    for index in CsclIndicesDescriptions:
        csclIndicesDescriptions[index.name] = index.value
    # End CSCL Descriptions

    # Participant Evolution
    participantEvolution = []
    importance = conv.graph.importance
    participantImportance = {}
    for participant in participant_list:
        participantImportance[participant.get_id()] = 0
    contributions = conv.get_contributions()
    for index, contribution in enumerate(contributions):
        for participant in participant_list:
            if participant == contribution.get_participant():
                participantImportance[participant.get_id()] += importance[contribution]
            nodeDict = {
                "nodeName": participant.get_id(),
                "x": index,
                "y": participantImportance[participant.get_id()]
            }
            participantEvolution.append(nodeDict)
    # End Participant Evolution

    # Social KB
    socialKB = []
    for index, contribution in enumerate(contributions):
        nodeDict = {
                "nodeName": "",
                "x": index,
                "y": 0
            }
        socialKB.append(nodeDict)
    # End Social KB

    csclDataDTO = CsclDataDTO(languageString, conceptMap, csclIndices, csclIndicesDescriptions, participantEvolution, participantInteractionGraph, socialKB)

    csclResponse = CsclResponse(
        csclDataDTO, "", True)
    try:
        jsonString = csclResponse.toJSON()
    except Exception as e:
        print("Error when serializing")
        raise e    
    return jsonString

def fileUploadPost():
    data = request.files['file']
    name = data.filename
    filename = data.filename
    serverFilename = str(time())
    if allowed_file(filename):
        filename = secure_filename(filename)
        path = os.path.join(UPLOAD_FOLDER, serverFilename)
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # path = os.path.join(path, filename)
        data.save(path)
        # path = os.path.join(path, filename)

    else:
        response = error("Invalid uploaded file!")
        return generate_response(response)
    response = success({"file": serverFilename})

    return generate_response(response)


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def success(data) -> Response:
    return jsonify({"data": data, "success": True, "errMsg": ""})

def error(message: str) -> Response:
    return jsonify({"data": {}, "success": False, "errMsg": message})


def generate_response(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Accept,Content-Type,Authorization,Access-Control-Allow-Origin,Cache-Control')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS,HEAD')
    return response, 200
