from enum import Enum

from rb.core.lang import Lang
from rb.core.document import Document
from rb_api.mass_customization.lesson import Lesson
from rb_api.mass_customization.lesson_keywords import LessonKeywords
from rb_api.mass_customization.lesson_themes import LessonThemes
from rb_api.mass_customization.lesson_expertise import LessonExpertise
import csv
import urllib.request
CSV_FILE_PATH = 'https://nextcloud.readerbench.com/index.php/s/bsKbYDrCSYFKNze/download'


class Annotators(Enum):
    NLP_PREPROCESSING = "NLP_PREPROCESSING"
    USE_BIGRAMS = "USE_BIGRAMS"
    SENTIMENT_ANALYSIS = "SENTIMENT_ANALYSIS"
    DIALOGISM = "DIALOGISM"
    TEXTUAL_COMPLEXITY = "TEXTUAL_COMPLEXITY"


class LessonReader:
    def __init__(self, lang: Lang, models: list, usePosTagging: bool, computeDialogism: bool, useBigrams: bool):
        self.lang = lang
        self.models = models

        self.usePosTagging = usePosTagging
        self.computeDialogism = computeDialogism
        self.useBigrams = useBigrams

        self.lessons = {}
        self.lessonDocuments = {}
        self.parse()

    def parse(self):
        content = urllib.request.urlopen(CSV_FILE_PATH).read().decode("utf-8")
        csv_reader = csv.DictReader(content.splitlines(), delimiter=";")

        next(csv_reader)
        for row in csv_reader:
            lesson = Lesson(row['title'], row['description'], row['mod'], row['unit'], row['lesson'], row['description'])
            # self.lessons.append(lesson)

            level1 = row['key_lvl1']
            level2 = row['key_lvl2']

            lesson_keywords = LessonKeywords(level1, level2)

            lesson.set_keywords(lesson_keywords)

            lesson_themes = LessonThemes(row['themes_sci'], row['themes_practice'], row['themes_guide'])

            lesson.set_themes(lesson_themes)

            lesson_expertise = LessonExpertise(row['exp_med_paedi'], row['exp_med_gyn'], row['exp_med_gp'], row['exp_med_other'],
                row['exp_nurse'], row['exp_nutrition'], row['exp_other'], row['exp_student'])

            lesson.set_expertise(lesson_expertise)

            pre = row['pre'] if row['pre'] else ""
            lesson.set_prerequisites(pre)

            post = row['post'] if row['post'] else ""
            lesson.set_postrequisites(post)

            lesson.set_time(row['time'])

            lesson.set_id(row['id'])
            self.lessons[str(lesson.lessonDescriptives)] = lesson


