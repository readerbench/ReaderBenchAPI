from flask import Flask, request, jsonify
from flask_cors import CORS
import rb_api.keywords.keywords as keywords
import rb_api.textual_complexity.textual_complexity as textual_complexity
import rb_api.amoc.amoc as amoc
import rb_api.text_similarity.text_similarity as text_similarity
from rb_api.text_extractor.universal_text_extractor import extract_raw_text
from werkzeug import secure_filename
import os
import uuid

app = Flask(__name__)
CORS(app)
 
@app.route("/api/v1/isalive")
def hello():
    return "Alive"

@app.route("/api/v1/keywords", methods=['OPTIONS'])
def keywordsOption():
    return keywords.keywordsOption()

@app.route("/api/v1/keywords", methods=['POST'])
def keywordsPost():
    return keywords.keywordsPost()

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

""" file should have proper extension, otherwise it will not work"""
@app.route('/api/v1/extract_text', methods=['POST'])
def extract_text():
    f = request.files['file']
    path_to_tmp_file = secure_filename(str(uuid.uuid4()) + f.filename)
    f.save(path_to_tmp_file)
    raw_text = extract_raw_text(path_to_tmp_file)
    try:
        os.remove(path_to_tmp_file)
    except OSError:
        pass
    return jsonify(raw_text)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006, threaded=True)
