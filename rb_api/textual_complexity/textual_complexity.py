from flask import Flask, request
import json
from rb.core.text_element import TextElement
from rb.core.lang import Lang
from rb.complexity.index_category import IndexCategory
from rb.complexity.complexity_index import compute_indices
from rb.core.document import Document

app = Flask(__name__)

def textualComplexityOption():
    return ""

def textualComplexityPost():
    params = json.loads(request.get_data())
    posTagging = params.get('pos-tagging')
    dialogism = params.get('dialogism')
    bigrams = params.get('bigrams')
    text = params.get('text')
    languageString = params.get('language')
    print(languageString)
    lang = Lang[languageString]
    lsa = params.get('lsa')
    lda = params.get('lda')
    w2v = params.get('w2v')
    threshold = params.get('threshold')

    textElement = Document(lang, text)
    # complexity = ComplexityIndex(lang, IndexCategory.SYNTAX, "syntax")
    processed = compute_indices(textElement)
    print(processed)
    return "Not dead"
