from flask import Flask
from flask_cors import CORS
import rest.keywords.keywords as keywords
import rest.textual_complexity as textual_complexity

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
 
if __name__ == "__main__":
    app.run()
