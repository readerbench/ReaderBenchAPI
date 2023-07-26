from enum import Enum


class JobTypeEnum(Enum):
    PIPELINE = 1
    CSCL = 2
    PREDICT = 3
    INDICES = 4
    OFFENSIVE = 5
    SENTIMENT = 6
    DIACRITICS = 7
    ANSWER_GEN = 8
    TEST_GEN = 9

class JobStatusEnum(Enum):
    PENDING = 1
    IN_PROGRESS = 2
    FINISHED = 3
    ERROR = 4