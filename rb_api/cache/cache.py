from cachelib import SimpleCache
from rb.similarity.vector_model import VectorModelType
from rb.similarity.word_vector_model import WordVectorModel
from rb.similarity.word2vec import Word2Vec
from rb.core.lang import Lang

cache = SimpleCache()


def get_model(vtype: VectorModelType, name: str, lang: Lang, size: int = 300) -> WordVectorModel:
    model_name = "{}_{}_{}_{}".format(vtype.name, name, lang.value, size)
    global cache
    model = cache.get(model_name)
    if model is None:
        if vtype == VectorModelType.WORD2VEC:
            model = Word2Vec(name, lang, size)
        else:
            model = WordVectorModel(vtype, name, lang, size)
        cache.set(model_name, model)
    return model


