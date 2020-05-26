from flask import Flask, request, jsonify
from rb.core.lang import Lang
from rb.similarity.lsa import LSA
from rb.similarity.lda import LDA
from rb.similarity.word2vec import Word2Vec

from rb_api.mass_customization.lesson_expertise import LessonExpertise
from rb_api.mass_customization.lesson_keywords import LessonKeywords
from rb_api.mass_customization.lesson_themes import LessonThemes
from .lesson_reader import LessonReader
from .lesson_reader_other import LessonReaderOther
from .jacc_similarity import JaccSimilarity
from rb.core.document import Document

import copy
import re
from rb_api.dto.mass_customization.mass_customization_response import MassCustomizationResponse

app = Flask(__name__)


MINUTES_PER_CME_POINT = 60.0
CORPUS = 'enea_tasa'
threshold = 0.589
threshold_other = 0.19
threshold_similarity = 0.05

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

def massCustomizationOption():
    return ""

def massCustomizationPost():
    json_request = request.json
    cme = json_request.get('cme')
    expertises = json_request.get('expertise')
    topics = json_request.get('topics')
    text = json_request.get('text')
    themes = json_request.get('themes')
    otherdomains = json_request.get('otherdomains')
    level = json_request.get('level')
    liked_lessons = json_request.get('likedlessons')
    disliked_lessons = json_request.get('dislikedlessons')

    lang = Lang.EN
    usePosTagging = True
    computeDialogism = False
    useBigrams = False

    lsa_corpora = LSA(CORPUS, lang)
    lda_corpora = LDA(CORPUS, lang)
    w2v_corpora = Word2Vec(CORPUS, lang)

    models = [lsa_corpora, lda_corpora, w2v_corpora]
    print(otherdomains)

    # Check if is other domains
    if otherdomains:
        lesson_reader_other = LessonReaderOther(lang, level, topics)
        all_lessons = lesson_reader_other.lessons
        kept_lessons = copy.deepcopy(all_lessons)

    else:
        lesson_reader = LessonReader(lang, models, usePosTagging, computeDialogism, useBigrams)
        all_lessons = lesson_reader.lessons

        kept_lessons = copy.deepcopy(all_lessons)

        #Filter lessons by expertise, topics and themes

        parsed_expertises = parse_expertise(expertises)

        kept_lessons = filter_by_expertise(kept_lessons, parsed_expertises)

        if not kept_lessons:
            return jsonify({
                "errorMsg": "No lessons matching your criteria were found. Please broaden your search."
            })

        parsed_topics = parse_topics(topics)
        kept_lessons = filter_by_topics(kept_lessons, parsed_topics)


        if not kept_lessons:
            return jsonify({
                "errorMsg": "No lessons matching your criteria were found. Please broaden your search."
            })

        parsed_themes = parse_themes(themes)

        if parsed_themes:
            kept_lessons = filter_by_themes(kept_lessons, parsed_themes)
        if not kept_lessons:
            return jsonify({
                "errorMsg": "No lessons matching your criteria were found. Please broaden your search."
            })

        #If CME is true, sum up credits of the remaining lessons (1 credit = 60 mins);

        has_sum_error = check_sum_lessons(kept_lessons, cme)

        if has_sum_error:
            return jsonify({
                "errorMsg": "The CME sum of the remaining lessons is less than 5. No lesson is retrieved!"
            })

    print(len(kept_lessons))
    #Compute semantic similarity between the free text and the remaining lessons
    #common for nutrition and other domains
    if text:
        kept_lessons = filter_by_similarity(kept_lessons, text, models, lang, otherdomains)
    if not kept_lessons:
        return jsonify({
            "errorMsg": "No lessons matching your criteria were found. Please broaden your search."
        })

    if otherdomains:
        #Verify liked and disliked lessons
        data = check_liked_disliked(kept_lessons, all_lessons, level, topics, liked_lessons, disliked_lessons)
    else:
        # Step 5: Return lessons in descending order by similarity score
        # Step 6: Append prerequisites and postrequisites
        data = build_dto(kept_lessons, all_lessons)

    mass_customization_response = MassCustomizationResponse(
        data, "", True)
    json_response = mass_customization_response.toJSON()

    return json_response

def parse_expertise(expertises):
    converted_expertises = set()
    for expertise in expertises:
        converted_expertises.add(EXPERTISE_TO_CONSTANT.get(expertise, 0))

    return converted_expertises


def filter_by_expertise(kept_lessons, expertises: set):
    aux_lessons = copy.deepcopy(kept_lessons)
    for lesson_descriptives, lesson in aux_lessons.items():

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
            del kept_lessons[str(lesson_descriptives)]

    return kept_lessons


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
            if topic.level1 == lesson_keywords.level1 and topic.level2 in lesson_keywords.level2:
                has_min_one_keyword = True
                break

        if not has_min_one_keyword:
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
        return False

    sum_credits = 0.0

    for _, lesson in kept_lessons.items():
        sum_credits += lesson.time / MINUTES_PER_CME_POINT

    if sum_credits < 5:
        return True

    return False


def filter_by_similarity(kept_lessons, text, models, lang, otherdomains):
    aux_lessons = copy.deepcopy(kept_lessons)
    print(aux_lessons)
    if otherdomains:
        for lesson in aux_lessons:
            is_similar = 0
            print(lesson['published_title'])
            for learn in lesson['learn_details']:
                document1 = Document(lang, text)
                document2 = Document(lang, learn)

                for vectorModel in models:
                    similarity_score = vectorModel.similarity(
                        document1, document2)

                    if similarity_score > threshold_other:
                        is_similar = 1

                print(similarity_score)
            if not is_similar:
                kept_lessons.remove(lesson)
    else:
        for lesson_descriptives, lesson in aux_lessons.items():
            document1 = Document(lang, text)
            document2 = Document(lang, lesson.description)
            is_similar = 0

            for vectorModel in models:
                similarity_score = vectorModel.similarity(
                    document1, document2)

                if similarity_score > threshold:
                    lesson.similarityScore = similarity_score
                    is_similar = 1

            if not is_similar:
                del kept_lessons[lesson_descriptives]
    print(len(kept_lessons))
    return kept_lessons


def check_liked_disliked(kept_lessons, all_lessons, level, topics, liked, disliked):
    data = dict()
    if liked or disliked:
        jacc_similarity = JaccSimilarity(topics, level)
        print(jacc_similarity)
        add_similar_lessons = []
        remove_similar_lessons = []

        for topic in topics:
            for like in liked:
                for lesson_name, lesson_sim in jacc_similarity.similarity[topic].items():
                    if like in lesson_sim and lesson_sim[like] > threshold_similarity:
                        add_similar_lessons.append(lesson_sim)
                for lesson_name, lesson_sim in jacc_similarity.similarity[topic][like].items():
                    if lesson_sim > threshold_similarity:
                        add_similar_lessons.append(lesson_name)

            for dislike in disliked:
                for lesson_name, lesson_sim in jacc_similarity.similarity[topic].items():
                    if dislike in lesson_sim and lesson_sim[dislike] > threshold_similarity:
                        remove_similar_lessons.append(lesson_sim)
                for lesson_name, lesson_sim in jacc_similarity.similarity[topic][dislike].items():
                    if lesson_sim > threshold_similarity:
                        remove_similar_lessons.append(lesson_name)

        for add_similar in add_similar_lessons:
            for lesson in all_lessons:
                if add_similar == lesson['published_title'] and lesson not in kept_lessons:
                    kept_lessons.append(lesson)

        for add_similar in remove_similar_lessons:
            for lesson in all_lessons:
                if add_similar == lesson['published_title'] and lesson in kept_lessons:
                    kept_lessons.remove(lesson)

        for lesson in kept_lessons:
            if lesson['published_title'] in disliked:
                kept_lessons.remove(lesson)

    data['lessons'] = list(kept_lessons)
    return data


def build_dto(kept_lessons, all_lessons):
    aux_lessons = copy.deepcopy(kept_lessons)
    cme_points = 0
    total_time = 0
    recommended = list()
    lessons = set()
    data = dict()

    for lesson_descriptives, lesson in aux_lessons.items():
        pre_requisites_ids = list()
        post_requisites_ids = list()
        total_time += int(lesson.time)
        recommended.append(lesson.id)

        if lesson.prerequisites:
            pre_array = lesson.prerequisites.split(',')
            for lesson_prerequisite in pre_array:
                if lesson_prerequisite in all_lessons:
                    prerequisite_lesson = all_lessons[lesson_prerequisite]
                    pre_requisites_ids.append(prerequisite_lesson.id)
                    lessons.add(prerequisite_lesson)
                    cme_points += int(prerequisite_lesson.time) / MINUTES_PER_CME_POINT
                    total_time += int(prerequisite_lesson.time)

        lesson.prerequisites = pre_requisites_ids

        if lesson.postrequisites:
            post_array = lesson.postrequisites.split(',')
            for lesson_postrequisite in post_array:
                if lesson_postrequisite in all_lessons:
                    postrequisite_lesson = all_lessons[lesson_postrequisite]
                    post_requisites_ids.append(postrequisite_lesson.id)
                    lessons.add(postrequisite_lesson)
                    cme_points += int(postrequisite_lesson.time) / MINUTES_PER_CME_POINT
                    total_time += int(postrequisite_lesson.time)

        lesson.postrequisites = post_requisites_ids
        lessons.add(lesson)

    data['cmePoints'] = cme_points
    data['recommended'] = recommended
    data['time'] = total_time
    data['lessons'] = list(lessons)

    return data
