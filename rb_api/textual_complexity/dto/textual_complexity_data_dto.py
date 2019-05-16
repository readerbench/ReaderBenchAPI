class TextualComplexityDataDTO:

    def __init__(self, lang, categoriesList, complexityIndices):
        self.lang = lang
        self.list = categoriesList
        self.complexityIndices = complexityIndices

    def __str__(self):
        return "TextualComplexityData (lang=%s):\nCategories=%s\nIndices=%s" % (self.lang, self.list, self.complexityIndices)
