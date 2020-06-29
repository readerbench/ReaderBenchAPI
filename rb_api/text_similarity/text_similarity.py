from flask import Flask, request
import json
from rb.utils.utils import str_to_lang
from rb.core.text_element import TextElement
from rb.core.lang import Lang
from rb.complexity.index_category import IndexCategory
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document
from rb.core.word import Word
from rb.similarity.vector_model import VectorModel, VectorModelType
from rb.similarity.lsa import LSA
from rb.similarity.lda import LDA
from rb.similarity.word2vec import Word2Vec
from rb_api.dto.text_similarity.text_similarity_response_dto import TextSimilarityResponse
from rb_api.dto.text_similarity.scores_dto import ScoresDTO
from rb_api.dto.text_similarity.score_dto import ScoreDTO
from rb_api.dto.text_similarity.pair_dto import PairDTO

app = Flask(__name__)


def textSimilarityOption():
    return ""


def textSimilarityPost():
    params = json.loads(request.get_data())
    corpus = params.get('corpus') if params.get(
        'corpus') != None else 'le_monde_small'
    languageString = params.get('language')
    lang = str_to_lang(languageString)
    texts = params.get('texts')

    vectorModels = []
    try:
        vectorModel = LSA(corpus, lang)
        vectorModels.append(vectorModel)
    except FileNotFoundError as inst:
        print(inst)

    try:
        vectorModel = LDA(corpus, lang)
        vectorModels.append(vectorModel)
    except FileNotFoundError as inst:
        print(inst)

    try:
        vectorModel = Word2Vec(corpus, lang)
        vectorModels.append(vectorModel)
    except FileNotFoundError as inst:
        print(inst)

    noTexts = len(texts)
    pairs = []
    for i in range(0, noTexts):
        document1 = Document(lang, texts[i])
        for j in range(i + 1, noTexts):
                document2 = Document(lang, texts[j])
                scores = []
                for vectorModel in vectorModels:
                        similarityScore = vectorModel.similarity(
                            document1, document2)
                        scoreDTO = ScoreDTO(vectorModel.type.name, similarityScore)
                        scores.append(scoreDTO)
                pairDTO = PairDTO(i, j, scores)
                pairs.append(pairDTO)

    # print(pairs)
    scoresDTO = ScoresDTO(lang, corpus, pairs)
    textSimilarityResponse = TextSimilarityResponse(scoresDTO, "", True)
    jsonString = textSimilarityResponse.toJSON()

    return jsonString
