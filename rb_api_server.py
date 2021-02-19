import json
import os
from rb_api.sentiment.sentiment_models import sentiment_post
import uuid

from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
from rb.utils.utils import str_to_lang
from werkzeug.utils import secure_filename

import rb_api.amoc.amoc as amoc
import rb_api.keywords.keywords as keywords
# import rb_api.diacritics.diacritics as diacritics
import rb_api.text_similarity.text_similarity as text_similarity
import rb_api.mass_customization.mass_customization as mass_customization
import rb_api.textual_complexity.textual_complexity as textual_complexity
import rb_api.feedback.feedback as feedback
from rb_api.cna.graph_extractor import compute_graph
from rb_api.cscl import cscl
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route("/api/v1/isalive")
def hello():
    return "Alive"


@app.route("/api/v1/keywords", methods=['OPTIONS'])
def keywordsOption():
   return keywords.keywordsOption()


@app.route("/api/v1/keywords", methods=['POST'])
def keywordsPost():
   return keywords.keywordsPost()

# @app.route("/api/v1/diacritics", methods=['POST'])
# def diacriticsPost():
    # return diacritics.diacriticsPost()


@app.route("/api/v1/textual-complexity", methods=['OPTIONS'])
def textualComplexityOption():
    return textual_complexity.textualComplexityOption()


@app.route("/api/v1/textual-complexity", methods=['POST'])
def textualComplexityPost():
    return textual_complexity.textualComplexityPost()


@app.route("/api/v1/amoc", methods=['OPTIONS'])
def amocOption():
    return amoc.amocOption()


@app.route("/api/v1/amoc", methods=['POST'])
def amocPost():
    return amoc.amocPost()


@app.route("/api/v1/text-similarity", methods=['OPTIONS'])
def textSimilarityOption():
    return text_similarity.textSimilarityOption()


@app.route("/api/v1/text-similarity", methods=['POST'])
def textSimilarityPost():
    return text_similarity.textSimilarityPost()

@app.route("/api/v1/mass-customization", methods=['OPTIONS'])
def massCustomizationOption():
    return mass_customization.massCustomizationOption()

@app.route("/api/v1/mass-customization", methods=['POST'])
def massCustomizationPost():
    return mass_customization.massCustomizationPost()


@app.route("/api/v1/cna-graph", methods=['OPTIONS'])
def computeCnaGraphOption():
    return ""


@app.route("/api/v1/cna-graph", methods=['POST'])
def computeCnaGraphPost():
    params = json.loads(request.get_data())
    texts = [doc["text"] for doc in params.get('texts')]
    languageString = params.get('lang')
    lang = str_to_lang(languageString)
    models = params.get('models')
    return compute_graph(texts, lang, models)


@app.route('/api/v1/extract_text', methods=['POST'])
def extract_text():
    """ file should have proper extension, otherwise it will not work"""
    from rb_api.text_extractor.universal_text_extractor import extract_raw_text
    f = request.files['file']
    path_to_tmp_file = secure_filename(str(uuid.uuid4()) + f.filename)
    f.save(path_to_tmp_file)
    raw_text = extract_raw_text(path_to_tmp_file)
    try:
        os.remove(path_to_tmp_file)
    except OSError:
        pass
    return jsonify(raw_text)

@app.route("/api/v1/cscl-processing", methods=['OPTIONS'])
def csclOption():
   return cscl.csclOption()


@app.route("/api/v1/cscl-processing", methods=['POST'])
def csclPost():
   return cscl.csclPost()


@app.route("/api/v1/file-upload", methods=['OPTIONS'])
def fileUploadOption():
   return cscl.csclOption()


@app.route('/api/v1/file-upload', methods=["POST"])
def fileUploadPost():
    return cscl.fileUploadPost()

@app.route("/api/v1/feedback", methods=['OPTIONS'])
def feedbackOption():
    return generate_response(success())


@app.route("/api/v1/feedback", methods=['POST'])
def feedbackPost():
    params = json.loads(request.get_data())
    text = params.get('text')
    response = dict()
    doc_indices = feedback.compute_textual_indices(text)
    response['feedback'] = feedback.automatic_feedback(doc_indices)
    response['score'] = feedback.automatic_scoring(doc_indices)
    response = success(response)
    return generate_response(response)

@app.route("/api/v1/sentiment", methods=['OPTIONS'])
def predict_sentiment_option():
    return ""

@app.route("/api/v1/sentiment", methods=['POST'])
def predict_sentiment_post():
    return sentiment_post(request)

def generate_response(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers',
                         'Origin, X-Requested-With, Accept,Content-Type,Authorization,Access-Control-Allow-Origin,Cache-Control')
    response.headers.add('Access-Control-Allow-Methods',
                         'GET,PUT,POST,DELETE,OPTIONS,HEAD')
    return response, 200


def success(data) -> Response:
    return jsonify({"data": data, "success": True, "errMsg": ""})


def error(message: str) -> Response:
    return jsonify({"data": {}, "success": False, "errMsg": message})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, debug=True)
