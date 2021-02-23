import json
import traceback
from pickle import NONE
from typing import Dict

from flask import Flask, jsonify, request
from rb import Lang
from rb.processings.diacritics.DiacriticsRestoration import \
    DiacriticsRestoration
from rb.processings.sentiment.SentimentAnalysis import SentimentAnalysis
from rb.utils.utils import str_to_lang
from rb_api.diacritics.diacriticsresponse import DiacriticsResponse
from rb_api.sentiment.sentiment_response import SentimentResponse


class SentimentModelsCache():

    INSTANCE = None

    def __init__(self):
        self.models: Dict[Lang, Dict[str, SentimentAnalysis]] = {}
    
    def get_model(self, lang: Lang, model_name: str):
        if lang not in self.models or model_name not in self.models[lang]:
            try:
                model = SentimentAnalysis(lang, model_name, 256)
                if lang not in self.models:
                    self.models[lang] = {}
                self.models[lang][model_name] = model
            except Exception as e:
                print(traceback.format_exc())
                return None
        return self.models[lang][model_name]
    
    @classmethod
    def get_instance(cls):
        if cls.INSTANCE is None:
            cls.INSTANCE = SentimentModelsCache()
        return cls.INSTANCE


def sentiment_post(request):
    params = json.loads(request.get_data())
    text = params.get("text")
    lang = str_to_lang(params.get("lang"))
    model_name = params["model"] if "model" in params else "base"
    model = SentimentModelsCache.get_instance().get_model(lang, model_name)
    if not model:
        return SentimentResponse(data="", errorMsg="Model doesn't exist", success=False).toJSON()
    prediction = model.process_text(text)
    return SentimentResponse(data={"prediction": prediction[0]}, errorMsg="", success=True).toJSON()
