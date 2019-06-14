from werkzeug.contrib.cache import MemcachedCache
from rb.similarity.vector_model import VectorModelType, VectorModel
from rb.core.lang import Lang

cache = MemcachedCache()

def get_model(type: VectorModelType, name: str, lang: Lang, size: int = 300):
    model_name = VectorModelType.name + "_" + name + "_" + lang.value + "_" + size
    global cache
    model = cache.get(model_name)
    if model is None:
        model = VectorModel(type, name, lang, size)
        cache.set(model_name, model)
    return model


