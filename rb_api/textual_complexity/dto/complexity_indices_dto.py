class ComplexityIndicesDTO:

    def __init__(self, category, valences):
        self.category = category
        self.valences = valences

    def __str__(self):
        return "ComplexityIndicesDTO: category=%s ; valences=%s\n" % (self.category, self.valences)
