import json
import os
from time import time

from flask import Flask, Response, jsonify, request
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
from rb.complexity.index_category import IndexCategory
from rb.core.cscl.community import Community
from rb.core.cscl.conversation import Conversation
from rb.core.cscl.cscl_indices import CsclIndices
from rb.core.cscl.cscl_indices_descriptions import CsclIndicesDescriptions
from rb.core.cscl.cscl_parser import load_from_xml
from rb.core.document import Document
from rb.core.lang import Lang
from rb.core.pos import POS
from rb.core.text_element import TextElement
from rb.processings.cscl.participant_evaluation import (evaluate_interaction,
                                                        evaluate_involvement,
                                                        evaluate_used_concepts,
                                                        perform_sna)
from rb.processings.keywords.keywords_extractor import KeywordExtractor
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import (VECTOR_MODELS,
                                                create_vector_model)
from rb.utils.utils import str_to_lang
from werkzeug.utils import secure_filename
from os import path

app = Flask(__name__)
ALLOWED_EXTENSIONS = set(['xml'])
UPLOAD_FOLDER = "upload"

def csclOption():
    return ""

def csclPost():
    params = json.loads(request.get_data())
    print(params)
    csclFile = params.get('cscl-file')
    languageString = params.get('language')
    lang = str_to_lang(languageString)

    print('Processing CSCL file ' + csclFile + ' using ' + lang.value + ' language')

    # conv_thread = Conversation.load_from_xml(lang, csclFile)
    basepath = path.dirname(__file__)
    filepath = path.abspath(path.join(basepath, "..", "..", "upload", csclFile))
    conv_thread = load_from_xml(lang, filepath)
    myCommunity = Community(lang=lang, container=None, community=conv_thread)
    fr_le_monde_word2vec = create_vector_model(lang, VectorModelType.from_str("word2vec"), "le_monde_small")
    myCommunity.graph = CnaGraph(docs=[myCommunity], models=[fr_le_monde_word2vec])

    conv = myCommunity.get_conversations()[0]
    print('Creating graph')
    conv.container.graph = CnaGraph(docs=[conv], models=[fr_le_monde_word2vec])
    print('Graph create')

    participant_list = conv.get_participants()  
    names = list(map(lambda p: p.get_id(), participant_list))

    csclData = {}

    # Begin Concept Map
    csclData["conceptMap"] = {}
    csclData["conceptMap"]["nodeList"] = []
    csclData["conceptMap"]["edgeList"] = []
    keywords_extractor = KeywordExtractor()
    # keywords = keywords_extractor.extract_keywords(text=conv.text, lang=lang, max_keywords=40, vector_model=fr_le_monde_word2vec)
    keywords = keywords_extractor.extract_keywords(text=conv.text, lang=lang)
    # print(keywords)
    for p in keywords:
        score = p[0]
        word = p[1]
        posStr = word.pos.value
        csclData["conceptMap"]["nodeList"].append(
            {
                "type": "Word",
                "uri": word.lemma + '_' + posStr,
                "displayName": word.lemma,
                "active": True,
                "degree": score
            }
        )
    for p in keywords:
        for q in keywords:
            posWord1 = p[1].pos.value
            posWord2 = q[1].pos.value
            csclData["conceptMap"]["edgeList"].append(
                {
                    "edgeType": "SemanticDistance",
                    "score": fr_le_monde_word2vec.similarity(p[1], q[1]),
                    "sourceUri": p[1].lemma + '_' + posWord1,
                    "targetUri": q[1].lemma + '_' + posWord2
                }
            )
    # End Concept Map

    # print('Participants are:')
    # print(names)

    evaluate_interaction(conv)
    # # conv.get_score(participant_list[0].get_id(), participant_list[1].get_id())
    evaluate_involvement(conv)
    evaluate_used_concepts(conv)
    perform_sna(conv, False)

    print('Finished computing indices')

    # Begin Participant Interaction Graph
    csclData["participantInteractionGraph"] = {}
    csclData["participantInteractionGraph"]["nodeList"] = []
    csclData["participantInteractionGraph"]["edgeList"] = []
    nameIndex = {}
    k = 0
    for n in names:
        csclData["participantInteractionGraph"]["nodeList"].append(
            {
                "type": "Author",
                "uri": k,
                "displayName": n,
                "active": True
            },
        )
        nameIndex[n] = k
        k += 1

    for n1 in names:
        for n2 in names:
            # print('Score for ' + n1 + ' ' + n2 + ' is:')
            # print(conv.get_score(n1, n2))
            csclData["participantInteractionGraph"]["edgeList"].append(
                {
                    "edgeType": "SyntacticDistance",
                    "score": conv.get_score(n1, n2),
                    "sourceUri": nameIndex[n1],
                    "targetUri": nameIndex[n2]
                },
            )
    # End Participant Interaction Graph

    # Begin CSCL Indices
    csclData["csclIndices"] = {}
    for p in participant_list:
        participantDict = {
            "INTER_ANIMATION_DEGREE": -1,
            "SOCIAL_KB": p.get_index(CsclIndices.SOCIAL_KB),
            "INDEGREE": p.get_index(CsclIndices.INDEGREE),
            "NO_CONTRIBUTION": p.get_index(CsclIndices.NO_CONTRIBUTION),
            "SCORE": p.get_index(CsclIndices.SCORE),
            "RHYTHMIC_COEFFICIENT": -1,
            "NO_VERBS":p.get_index(CsclIndices.NO_VERBS),
            "FREQ_MAX_RHYTMIC_INDEX": -1,
            "RHYTHMIC_INDEX": -1,
            "PERSONAL_REGULARITY_ENTROPY": -1,
            "OUTDEGREE": p.get_index(CsclIndices.OUTDEGREE),
            "NO_NOUNS": p.get_index(CsclIndices.NO_NOUNS)
        }
        csclData["csclIndices"][p.get_id()] = participantDict
    # End CSCL Indices

    # Begin CSCL Descriptions
    csclData["csclIndicesDescription"] = {}
    for index in CsclIndicesDescriptions:
        csclData["csclIndicesDescription"][index.name] = index.value
    # End CSCL Descriptions

    # print (csclData)
    response = success(csclData)

    return response

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
