from enum import Enum, unique

@unique
class TwoModeGraphNodeTypeDTO(Enum):
    AUTHOR = 'Author'
    ARTICLE = 'Article'
    USER_QUERY = 'UserQuery'
    WORD = 'Word'
    INFERRED = 'Inferred'
    TEXT_BASED = 'TextBased'