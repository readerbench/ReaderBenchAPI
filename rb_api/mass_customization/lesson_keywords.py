class LessonKeywords:

    def __init__(self, level1: str, level2: str):
        self.level1 = level1
        self.level2 = level2

    def __str__(self):
        return "LessonKeywords:\nLevel1:%s\nLevel2:%s" % (self.level1, self.level2)
