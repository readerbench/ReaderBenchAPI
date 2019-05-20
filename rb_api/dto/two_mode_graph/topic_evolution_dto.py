from rb_api.dto.two_mode_graph.article_keyword_dto import ArticleKeywordDTO

class TopicEvolutionDTO():

    def __init__(self):
        self.word_list = []
        self.year_list = []


    def add_year(self, year: int) -> None:
        self.year_list.append(year)

    
    def add_keyword(self, value: str, score: float) -> None:
        word_values = [word for word in self.word_list if word.value == value]
        if word_values:
            word_values[0].score_list.append(score) # there is only 1 word for value
            return

        keyword = ArticleKeywordDTO(value, "Keyword")
        keyword.score_list.append(score)
        self.word_list.append(keyword)

    
    def normalize(self):
        for year_index, _ in enumerate(self.year_list):
            maxim = max([word.score_list[year_index] for word in self.word_list])

            if maxim > 0:
                for word in self.word_list:
                    word.score_list[year_index] = word.score_list[year_index] / maxim
            