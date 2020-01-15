from rb.core.lang import Lang
from rb_api.mass_customization.lesson import Lesson
from rb_api.mass_customization.lesson_keywords import LessonKeywords
from rb_api.mass_customization.lesson_themes import LessonThemes
from rb_api.mass_customization.lesson_expertise import LessonExpertise
import csv

class LessonReader:

    CSV_FILE_PATH = 'https://nextcloud.readerbench.com/index.php/s/bsKbYDrCSYFKNze/download'

    def __init__(self, lang: str, lsaCorpora: str, ldaCorpora: str, w2vCorpora: str, usePosTagging: bool, computeDialogism: bool, useBigrams: bool):
        self.lang = lang
        # load semantic models
        self.usePosTagging = usePosTagging
        self.computeDialogism = computeDialogism
        self.useBigrams = useBigrams
        lessons = {}
        lessonDocuments = {}

    def parse():
        with open(CSV_FILE_PATH) as csv_file:  
            csv_reader = csv.DictReader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    pass
                else:
                    lesson = Lesson(row['title'], row['description'], row['mod'], row['unit'], row['lesson'], row['description'])

                    level1 = row['key_lvl1']
                    level2 = row['key_lvl2']
                    keywords = LessonKeywords(level1, level2)
                    
                    themes = LessonThemes(row['themes_sci'], row['themes_practice'], row['themes_guide'])
                    
                    expertise = LessonExpertise(row['exp_med_paedi'], row['exp_med_gyn'], row['exp_med_gp'], row['exp_med_other'],
                        row['exp_nurse'], row['exp_nutrition'], row['exp_other'], row['exp_student'])
                    

                    row['pre']
                    row['post']

                    row['time']
                    row['url']
                
                line_count += 1