class LessonExpertise:

    def __init__(self, medicinePaeditrician: bool, medicineGynocologist: bool, medicineGp: bool, medicineOther: bool, nursing: bool, nutrition: bool, other: bool, student: bool):
        self.medicinePaeditrician = medicinePaeditrician
        self.medicineGynocologist = medicineGynocologist
        self.medicineGp = medicineGp
        self.medicineOther = medicineOther
        self.nursing = nursing
        self.nutrition = nutrition
        self.other = other
        self.student = student

    def __str__(self):
        return "LessonExpertise: (medicinePaeditrician: %s) (medicineGynocologist: %s) (medicineGp: %s) (medicineOther: %s) (nursing: %s) (nutrition: %s) (other: %s) (student: %s)" % (self.medicinePaeditrician, self.medicineGynocologist, self.medicineGp, self.medicineOther, self.nursing, self.nutrition, self.other, self.student)
