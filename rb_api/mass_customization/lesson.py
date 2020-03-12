from rb_api.mass_customization.lesson_descriptives import LessonDescriptives


class Lesson:

    def __init__(self, title: str, description: str, module: int, unit: int, lesson: int, help: str):
        self.title = title
        self.description = description
        self.lessonDescriptives = LessonDescriptives(module, unit, lesson)
        self.similarityScore = 0.0
        self.help = help

    def set_keywords(self, keywords):
        self.keywords = keywords

    def set_themes(self, themes):
        self.themes = themes

    def set_expertise(self, lesson_expertise):
        self.lesson_expertise = lesson_expertise

    def set_prerequisites(self, pre):
        self.prerequisites = pre

    def set_postrequisites(self, post):
        self.postrequisites = post

    def set_time(self, time):
        self.time = time

    def set_id(self, id):
        self.id = id
