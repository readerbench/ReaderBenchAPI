from flask import Flask
from flask_cors import CORS
import rb_api.keywords.keywords as keywords
import rb_api.textual_complexity.textual_complexity as textual_complexity
import rb_api.amoc.amoc as amoc


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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006)
