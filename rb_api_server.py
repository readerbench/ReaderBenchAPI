from flask import Flask
from flask_cors import CORS
import rb_api.keywords.keywords as keywords
import rb_api.textual_complexity.textual_complexity as textual_complexity
import rb_api.amoc.amoc as amoc
import rb_api.text_similarity.text_similarity as text_similarity
import rb_api.mass_customization.mass_customization as mass_customization


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

@app.route("/api/v1/mass-customization", methods=['OPTIONS'])
def massCustomizationOption():
    return mass_customization.massCustomizationOption()

@app.route("/api/v1/mass-customization", methods=['POST'])
def massCustomizationPost():
    return mass_customization.massCustomizationPost()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006)
