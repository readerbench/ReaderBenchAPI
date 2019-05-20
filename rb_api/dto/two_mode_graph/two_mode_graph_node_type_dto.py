from enum import Enum, unique

@unique
class TwoModeGraphNodeTypeDTO(Enum):
    AUTHOR = 'author'
    ARTICLE = 'article'
    USER_QUERY = 'user_query'
    WORD = 'word'
    INFERRED = 'inferred'
    TEXT_BASED = 'text_based'