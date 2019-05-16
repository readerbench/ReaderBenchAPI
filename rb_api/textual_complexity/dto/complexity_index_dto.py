class ComplexityIndexDTO:

    def __init__(self, index, value):
        self.index = index
        self.value = value
    
    def __str__(self):
        return "ComplexityIndexDTO: index=%s ; value=%s\n" % (self.index, self.value)
