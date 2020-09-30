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
from rb.processings.cscl.participant_evaluation import evaluate_interaction, evaluate_involvement, evaluate_textual_complexity, get_block_importance, perform_sna
from rb.processings.keywords.keywords_extractor import extract_keywords
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import (VECTOR_MODELS,
                                                create_vector_model)
from rb.utils.utils import str_to_lang
from werkzeug.utils import secure_filename

from rb_api.cna.graph_extractor import compute_graph_cscl
from rb_api.dto.cscl.cscl_data_dto import CsclDataDTO
from rb_api.dto.cscl.cscl_response import CsclResponse
from rb.core.text_element_type import TextElementType

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
    lsaCorpus = params.get('lsa')
    ldaCorpus = params.get('lda')
    word2vecCorpus = params.get('w2v')

    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "..", "upload", csclFile))
    conv_thread = load_from_xml(filepath)
    conv = Conversation(lang=lang, conversation_thread=conv_thread, apply_heuristics=False)
    vectorModels = []
    if not "".__eq__(lsaCorpus):
        vectorModels.append(create_vector_model(lang, VectorModelType.from_str("lsa"), lsaCorpus))
    if not "".__eq__(ldaCorpus):
        vectorModels.append(create_vector_model(lang, VectorModelType.from_str("lda"), ldaCorpus))
    if not "".__eq__(word2vecCorpus):
        vectorModels.append(create_vector_model(lang, VectorModelType.from_str("word2vec"), word2vecCorpus))
    conv.graph = CnaGraph(docs=[conv], models=vectorModels)

    participant_list = conv.get_participants()  
    names = [p.get_id() for p in participant_list]
    
    conceptMaps = {
        'LSA': None,
        'LDA': None,
        'WORD2VEC': None
    }
    # Begin Concept Map
    for vectorModel in vectorModels:
        keywords = extract_keywords(text=conv.text, lang=lang, vector_model=vectorModel)
        conceptMap = {
            "nodeList": [],
            "edgeList": [],
        }
        for score, word in keywords:
            conceptMap["nodeList"].append(
                {
                    "type": "Word",
                    "uri": word,
                    "displayName": word,
                    "active": True,
                    "degree": score
                }
            )
        vectors = {}
        for _, keyword in keywords:
            vectors[keyword] = vectorModel.get_vector(keyword)
        for _, keyword1 in keywords:
            for _, keyword2 in keywords:
                conceptMap["edgeList"].append(
                    {
                        "edgeType": "SemanticDistance",
                        "score": vectorModel.similarity(vectors[keyword1], vectors[keyword2]),
                        "sourceUri": keyword1,
                        "targetUri": keyword2
                    }
                )
        conceptMaps[vectorModel.type.name] = conceptMap
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
    for i, p in enumerate(participant_list):
        participantInteractionGraph["nodeList"].append(
            {
                "type": "Author",
                "uri": i,
                "displayName": p.get_id(),
                "active": True,
                "degree": p.get_index(CNAIndices.INDEGREE) + p.get_index(CNAIndices.OUTDEGREE)
            },
        )
        nameIndex[p.get_id()] = i
        
    for p1 in participant_list:
        for p2 in participant_list:
            participantInteractionGraph["edgeList"].append(
                {
                    "edgeType": "SemanticDistance",
                    "score": conv.get_score(p1.get_id(), p2.get_id()),
                    "sourceUri": nameIndex[p1.get_id()],
                    "targetUri": nameIndex[p2.get_id()]
                },
            )
    # End Participant Interaction Graph

    # Begin CSCL Indices
    csclIndices = {}

    contributions = conv.get_contributions()
    noParticipantContributions = {}
    for index, p in enumerate(participant_list):
        noParticipantContributions[p.get_id()] = 0
    for index, contribution in enumerate(contributions):
        noParticipantContributions[contribution.get_participant().get_id()] += 1

    for p in participant_list:
        # adunat social kb din contributiile lui
        participantDict = {
            "CONTRIBUTIONS_SCORE": p.get_index(CNAIndices.CONTRIBUTIONS_SCORE),
            # "INTERACTION_SCORE": p.get_index(CNAIndices.INTERACTION_SCORE),
            "SOCIAL_KB": p.get_index(CNAIndices.SOCIAL_KB),
            "OUTDEGREE": p.get_index(CNAIndices.OUTDEGREE),
            "INDEGREE": p.get_index(CNAIndices.INDEGREE),
            "NO_CONTRIBUTIONS": noParticipantContributions[p.get_id()],
            "CLOSENESS": p.get_index(CNAIndices.CLOSENESS),
            "BETWEENNESS": p.get_index(CNAIndices.BETWEENNESS),
            "EIGENVECTOR": p.get_index(CNAIndices.EIGENVECTOR),
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
    
    for index, contribution in enumerate(contributions):
        for participant in participant_list:
            if participant == contribution.get_participant():
                participantImportance[participant.get_id()] += importance[contribution] # suma muchiilor - de luat in core
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
        socialKB.append(0)

    for index1, contribution1 in enumerate(contributions):
        for index2, contribution2 in enumerate(contributions[:index1]):
            weight = get_block_importance(conv.graph.filtered_graph, contribution1, contribution2)
            if weight > 0 and contribution1.get_participant() != contribution2.get_participant():
                socialKB[index1] += weight

    socialKBResponse = []
    for index, contribution in enumerate(contributions):
        nodeDict = {
                "nodeName": "",
                "x": index,
                "y": socialKB[index]
            }
        socialKBResponse.append(nodeDict)
    # End Social KB

    # Tabel dupa replici; pt fiecare replica afisam social kb, local importance, total importance
    sumImportance = 0
    sumKB = 0
    contributionsIndices = {
        'contributions': [],
        'total': {
            'SOCIAL_KB': 0,
            'LOCAL_IMPORTANCE': 0
        }
    }
    for index, contribution in enumerate(contributions):
        sumKB += socialKB[index]
        sumImportance += importance[contribution]
        rawContrib = contribution.get_raw_contribution()
        contributionDict = {
            "participant": contribution.get_participant().get_id(),
            "genid": contribution.get_raw_contribution()['id'],
            "ref": contribution.get_raw_contribution()['parent_id'],
            "timestamp": contribution.get_timestamp().strftime('%Y-%m-%d %H:%M:%S.%f %Z'),
            "text": contribution.get_raw_contribution()['text'],
            "SOCIAL_KB": socialKB[index],
            "LOCAL_IMPORTANCE": importance[contribution],
        }
        contributionsIndices['contributions'].append(contributionDict)
    contributionsIndices['total'] = {
        "SOCIAL_KB": sumKB,
        "LOCAL_IMPORTANCE": sumImportance,
    }

    contibutionsTexts = [contribution.get_raw_contribution()['text'] for contribution in contributions]
    cnaModels = []
    for model in vectorModels:
        cnaModel = {
            'corpus': model.corpus,
            'model': model.type.name.lower()
        }
        cnaModels.append(cnaModel)
    textLabels = [
        'Utterance',
        'Sentence'
    ]
    cnaGraph = compute_graph_cscl(texts=contibutionsTexts, lang=lang, models=cnaModels, textLabels=textLabels)

    csclDataDTO = CsclDataDTO(languageString, conceptMaps, csclIndices, csclIndicesDescriptions, participantEvolution, participantInteractionGraph, socialKBResponse, contributionsIndices, cnaGraph)

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
