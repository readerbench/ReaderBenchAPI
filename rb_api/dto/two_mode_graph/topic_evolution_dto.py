from rb_api.dto.two_mode_graph.article_keyword_dto import ArticleKeywordDTO
from rb_api.json_serialize import JsonSerialize

import json

class TopicEvolutionDTO(JsonSerialize):

    def __init__(self):
        self.wordList = []
        self.yearList = []


    def add_year(self, year: int) -> None:
        self.yearList.append(year)

    
    def add_keyword(self, value: str, score: float) -> None:
        word_values = [word for word in self.wordList if word.value == value]
        if word_values:
            word_values[0].score_list.append(score) # there is only 1 word for value
            return

        keyword = ArticleKeywordDTO(value, "Keyword")
        keyword.scoreList.append(score)
        self.wordList.append(keyword)

    
    def normalize(self):
        for year_index, _ in enumerate(self.yearList):
            maxim = max([word.score_list[year_index] for word in self.wordList])

            if maxim > 0:
                for word in self.wordList:
                    word.score_list[year_index] = word.score_list[year_index] / maxim
            

    def serialize(self):
        return json.dumps(self.__dict__)
