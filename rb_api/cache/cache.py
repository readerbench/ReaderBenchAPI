from werkzeug.contrib.cache import MemcachedCache
from rb.similarity.vector_model import VectorModelType, VectorModel
from rb.core.lang import Lang

cache = MemcachedCache()

def get_model(vtype: VectorModelType, name: str, lang: Lang, size: int = 300):
    model_name = "{}_{}_{}_{}".format(vtype.name, name, lang.value, size)
    global cache
    model = cache.get(model_name)
    if model is None:
        model = VectorModel(vtype, name, lang, size)
        cache.set(model_name, model)
    return model


