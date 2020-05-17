class LessonDescriptives:

    def __init__(self, module: int, unit: int, lesson: int):
        self.module = module
        self.unit = unit
        self.lesson = lesson

    def __str__(self):
        return "%s.%s.%s" % (self.module, self.unit, self.lesson)
