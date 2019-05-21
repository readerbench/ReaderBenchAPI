from rb.core.lang import Lang
from rb_api.amoc.comprehension_model_service import ComprehensionModelService
from rb.similarity.word2vec import Word2Vec

if __name__ == "__main__":
    semantic_models = [Word2Vec('coca', Lang.EN)]
    cms = ComprehensionModelService(semantic_models, Lang.EN,
                                    0.3, 20, 5)

    text = "A young knight rode through the forest. The knight was unfamiliar with the country. Suddenly, a dragon appeared. The dragon was kidnapping a beautiful princess. The knight wanted to free the princess. The knight wanted to marry the princess. The knight hurried after the dragon. They fought for life and death. Soon, the knight's armor was completely scorched. At last, the knight killed the dragon. The knight freed the princess. The princess was very thankful to the knight. She married the knight."
    result = cms.run(text)

    print(result.toJSON())