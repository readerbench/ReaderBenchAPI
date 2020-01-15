class LessonThemes:

    def __init__(self, theory: bool, practice: bool, guidelines: bool):
        self.theory = theory
        self.practice = practice
        self.guidelines = guidelines

    def __str__(self):
        return "LessonThemes: (theory: %s) (practice: %s) (guidelines: %s)" % (self.theory, self.practice, self.guidelines)
