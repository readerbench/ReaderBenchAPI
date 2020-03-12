from flask import Flask, request, jsonify
from rb.core.lang import Lang
from rb.similarity.lsa import LSA
from rb.similarity.lda import LDA
from rb.similarity.word2vec import Word2Vec

from rb_api.mass_customization.lesson_expertise import LessonExpertise
from rb_api.mass_customization.lesson_keywords import LessonKeywords
from rb_api.mass_customization.lesson_themes import LessonThemes
from .lesson_reader import LessonReader
from rb.core.document import Document
import json
import csv
import copy
import re

from rb_api.dto.text_similarity.pair_dto import PairDTO
from rb_api.dto.text_similarity.score_dto import ScoreDTO
from rb_api.dto.text_similarity.scores_dto import ScoresDTO
from rb_api.dto.text_similarity.text_similarity_response_dto import TextSimilarityResponse

app = Flask(__name__)


MINUTES_PER_CME_POINT = 60.0
CORPUS = 'enea_tasa'

class Constants:
    EXPERTISE_MED_PAEDI = 1
    EXPERTISE_MED_GYNO = 2
    EXPERTISE_MED_GP = 3
    EXPERTISE_MED_OTHER = 4
    EXPERTISE_NURSE = 5
    EXPERTISE_NUTRI = 6
    EXPERTISE_OTHER = 7
    EXPERTISE_STUDENT = 8

    THEME_SCIENCE = 1
    THEME_GUIDELINES = 2
    THEME_PRACTICE = 3


EXPERTISE_TO_CONSTANT = {
    "medicine_paediatrician":
     Constants.EXPERTISE_MED_PAEDI,
    "medicine_gynocologist":
         Constants.EXPERTISE_MED_GYNO,
    "medicine_gp":
         Constants.EXPERTISE_MED_GP,
    "medicine_other":
         Constants.EXPERTISE_MED_OTHER,
    "nursing":
         Constants.EXPERTISE_NURSE,
    "nutrition":
         Constants.EXPERTISE_NUTRI,
    "other":
         Constants.EXPERTISE_OTHER,
    "student":
         Constants.EXPERTISE_STUDENT,
}

THEME_TO_CONSTANT = {
    "science": Constants.THEME_SCIENCE,
    "guidelines": Constants.THEME_GUIDELINES,
    "practice": Constants.THEME_PRACTICE
}


def massCustomizationPost():
    json_request = request.json
    cme = json_request.get('cme')
    expertises = json_request.get('expertise')
    topics = json_request.get('topics')
    text = json_request.get('text')
    themes = json_request.get('themes')

    lang = Lang.EN
    usePosTagging = True
    computeDialogism = False
    useBigrams = False

    lsa_corpora = LSA(CORPUS, lang)
    lda_corpora = LDA(CORPUS, lang)
    w2v_corpora = Word2Vec(CORPUS, lang)

    models = [lsa_corpora, lda_corpora, w2v_corpora]

    lesson_reader = LessonReader(lang, models, usePosTagging, computeDialogism, useBigrams)

    all_lessons = lesson_reader.lessons

    kept_lessons = copy.deepcopy(all_lessons)

    # Step 1: Filter lessons by expertise, topics and themes

    parsed_expertises = parse_expertise(expertises)

    kept_lessons = filter_by_expertise(kept_lessons, parsed_expertises)

    if not kept_lessons:
        return jsonify({
            "error": "No lessons matching your criteria were found. Please broaden your search."
        })

    parsed_topics = parse_topics(topics)

    kept_lessons = filter_by_topics(kept_lessons, parsed_topics)

    if not kept_lessons:
        return jsonify({
            "error": "No lessons matching your criteria were found. Please broaden your search."
        })

    parsed_themes = parse_themes(themes)

    if parsed_themes:
        kept_lessons = filter_by_themes(kept_lessons, parsed_themes)
    if not kept_lessons:
        return jsonify({
            "error": "No lessons matching your criteria were found. Please broaden your search."
        })

    # Step 2: If CME is true, sum up credits of the remaining lessons (1 credit = 60 mins);

    has_sum_error = check_sum_lessons(kept_lessons, cme)

    if has_sum_error:
        return jsonify({
            "error": "The CME sum of the remaining lessons is less than 5. No lesson is retrieved!"
        })

    # Step 3: Compute semantic similarity between the free text and the remaining lessons

    #other_topics = text

    if text:
        kept_lessons = filter_by_similarity(kept_lessons, text, models, lang)
    if not kept_lessons:
        return jsonify({
            "error": "No lessons matching your criteria were found. Please broaden your search."
        })


def parse_expertise(expertises):
    converted_expertises = set()
    for expertise in expertises:
        converted_expertises.add(EXPERTISE_TO_CONSTANT.get(expertise, 0))

    return converted_expertises


def filter_by_expertise(kept_lessons, expertises: set):
    aux_lessons = copy.deepcopy(kept_lessons)

    for key, value in aux_lessons.items():
        lesson_descriptives = key
        lesson = value

        lesson_expertise: LessonExpertise = lesson.lesson_expertise

        if not(
                (Constants.EXPERTISE_MED_GP in expertises and lesson_expertise.medicineGp) or
                (Constants.EXPERTISE_MED_GYNO in expertises and lesson_expertise.medicineGynocologist) or
                (Constants.EXPERTISE_MED_PAEDI in expertises and lesson_expertise.medicinePaeditrician) or
                (Constants.EXPERTISE_MED_OTHER in expertises and lesson_expertise.medicineOther) or
                (Constants.EXPERTISE_NURSE in expertises and lesson_expertise.nursing) or
                (Constants.EXPERTISE_NUTRI in expertises and lesson_expertise.nutrition) or
                (Constants.EXPERTISE_OTHER in expertises and lesson_expertise.nutrition) or
                (Constants.EXPERTISE_STUDENT in expertises and lesson_expertise.student)
        ):
            del kept_lessons[lesson_descriptives]

    return aux_lessons


def parse_topics(topics) -> list:
    converted_topics = list()

    for topic in topics:
        dict_obj: dict = topic
        for key, value in dict_obj.items():
            keyword_level_1 = key
            keyword_level_2: list = value
            for keyword in keyword_level_2:
                level_1 = convert(keyword_level_1)
                level_2 = convert(keyword)
                lesson_keyword = LessonKeywords(level_1, level_2)
                converted_topics.append(lesson_keyword)
    return converted_topics


def convert(keyword):
    level = re.sub(r'[^a-zA-Z\s-]', " ", keyword)
    level: str = re.sub(r' +', " ", level)
    level.strip().lower()

    return level


def filter_by_topics(kept_lessons, parsed_topics):
    aux_lessons = copy.deepcopy(kept_lessons)

    for lesson_descriptives, lesson in aux_lessons.items():
        lesson_keywords = lesson.keywords

        has_min_one_keyword = False

        for topic in parsed_topics:
            if topic.level_1 == lesson_keywords.level_1 and topic.level2 in lesson_keywords.level_2:
                has_min_one_keyword = True
                break

        if has_min_one_keyword:
            print(f"Lesson {lesson_descriptives}  has no matching keywords (lesson keywords: {lesson.keywords})")
            del kept_lessons[lesson_descriptives]

    return kept_lessons


def parse_themes(themes):
    parsed_themes = set()

    for theme in themes:
        parsed_themes.add(THEME_TO_CONSTANT.get(theme.lower(), 0))

    return parsed_themes


def filter_by_themes(kept_lessons, parsed_themes):
    aux_lessons = copy.deepcopy(kept_lessons)

    for lesson_descriptives, lesson in aux_lessons.items():
        lesson_themes: LessonThemes = lesson.themes
        if not (
                (Constants.THEME_SCIENCE in parsed_themes and lesson_themes.theory) or
                (Constants.THEME_GUIDELINES in parsed_themes and lesson_themes.guidelines) or
                (Constants.THEME_PRACTICE in parsed_themes and lesson_themes.practice)
        ):
            del kept_lessons[lesson_descriptives]

    return kept_lessons


def check_sum_lessons(kept_lessons, cme):
    if not cme:
        return True

    sum_credits = 0.0

    for _, lesson in kept_lessons.items():
        sum_credits += lesson.time / MINUTES_PER_CME_POINT

    if sum_credits < 5:
        return True

    return False


def filter_by_similarity(kept_lessons, text, models, lang):
    aux_lessons = copy.deepcopy(kept_lessons)


    for lesson_descriptives, lesson in aux_lessons.items():
        no_texts = len(text)
        pairs = []
        for i in range(0, no_texts):
            document1 = Document(lang, text[i])
            for j in range(i + 1, no_texts):
                document2 = Document(lang, text[j])
                scores = []
                for vectorModel in models:
                    similarity_score = vectorModel.similarity(
                        document1, document2)
                    score_dto = ScoreDTO(vectorModel.type.name, similarity_score)
                    scores.append(score_dto)
                pair_dto = PairDTO(i, j, scores)
                pairs.append(pair_dto)

        print(pairs)
        scores_dto = ScoresDTO(lang, CORPUS, pairs)


    return kept_lessons










def massCustomizationPost1():
    params = request.json
    print(params)
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



    noLessons = len(lessons)
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

    return ""

def checkMissingParameters(request):
    params = json.loads(request.get_data())
    expertise = params.get('expertise')
    topics = params.get('topics')
    themes = params.get('themes')

