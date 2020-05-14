from flask import Flask, request, jsonify
import json

from rb_api.diacritics.diacriticsresponse import DiacriticsResponse
from rb.processings.diacritics.DiacriticsRestoration import DiacriticsRestoration

app = Flask(__name__)

def diacriticsPost():
    params = json.loads(request.get_data())
    text = params.get('text')
    mode = params.get('mode')
    diacriticsrestoration = DiacriticsRestoration()

    try:
        restorationresult = diacriticsrestoration.process_string(text, mode=mode)
        return DiacriticsResponse(data=restorationresult, errorMsg="", success=True).toJSON()
    except Exception as e:
        return DiacriticsResponse(data="", errorMsg=str(e), success=False).toJSON()
