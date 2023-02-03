from enum import Enum


class JobTypeEnum(Enum):
    PIPELINE = 1
    CSCL = 2

class JobStatusEnum(Enum):
    PENDING = 1
    IN_PROGRESS = 2
    FINISHED = 3
    ERROR = 4