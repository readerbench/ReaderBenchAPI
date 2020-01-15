from flask import Flask, request
from rb.core.lang import Lang
from rb.similarity.lsa import LSA
from rb.similarity.lda import LDA
from rb.similarity.word2vec import Word2Vec
import json
import csv

app = Flask(__name__)

def massCustomizationOption():
    return ""


def massCustomizationPost():
    params = json.loads(request.get_data())
    cme = params.get('cme')
    expertise = params.get('expertise')
    topics = params.get('topics')
    text = params.get('text')
    themes = params.get('themes')

    lang = Lang.EN
    usePosTagging = True
    computeDialogism = False
    useBigrams = False
    corpus = 'enea_tasa'

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

    # read lessons file here

    # noLessons = len(lessons)
    # pairs = []
    # for i in range(0, noTexts):
    #     document1 = Document(lang, texts[i])
    #     for j in range(i + 1, noTexts):
    #             document2 = Document(lang, texts[j])
    #             scores = []
    #             for vectorModel in vectorModels:
    #                     similarityScore = vectorModel.similarity(
    #                         document1, document2)
    #                     scoreDTO = ScoreDTO(vectorModel.type.name, similarityScore)
    #                     scores.append(scoreDTO)
    #             pairDTO = PairDTO(i, j, scores)
    #             pairs.append(pairDTO)

    # # print(pairs)
    # scoresDTO = ScoresDTO(lang, corpus, pairs)
    # textSimilarityResponse = TextSimilarityResponse(scoresDTO, "", True)
    # jsonString = textSimilarityResponse.toJSON()

    return ""

def checkMissingParameters(request):
    params = json.loads(request.get_data())
    expertise = params.get('expertise')
    topics = params.get('topics')
    themes = params.get('themes')

    