from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
from rb.core.document import Document
from rb.core.lang import Lang
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.cna.cna_graph import CnaGraph
from rb.complexity.complexity_index import compute_indices
import ast
import numpy as np


class RussianAvsB():
    
    def __init__(self, model_path='AvsB_ru', rb_indices_path='ru_indices.list', bert_model_name='DeepPavlov/rubert-base-cased'):
        self.tokenizer, self.bert = self.load_rubert(bert_model_name)
        self.model = self.load_model(model_path)
        self.ru_word2vec = self.load_ru_word2vec()
        self.rb_indices_order = self.load_indices_list(rb_indices_path)
        
    
    def load_rubert(self, bert_model_name):
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        bert = TFAutoModel.from_pretrained(bert_model_name, from_pt=True)
        return tokenizer, bert
    
    
    def load_model(self, path):
        model = tf.keras.models.load_model(path)
        return model
    
    
    def load_ru_word2vec(self):
        ru_word2vec = create_vector_model(Lang.RU, VectorModelType.from_str('word2vec'), 'muse')
        return ru_word2vec
    
    
    def load_indices_list(self, path):
        with open(path, 'r') as f:
            x = f.read()
            x.strip()
            return ast.literal_eval(x)
        
    def get_bert_embedding(self, text):
        processed = self.tokenizer(text, padding=True, return_tensors='np')
        bert_output = self.bert(input_ids=processed['input_ids'], attention_mask=processed['attention_mask'], token_type_ids=processed['token_type_ids'])
        return np.ndarray.flatten(np.mean(bert_output.last_hidden_state, axis=1))
    
    
    def normalize_rb(self, input_rb):
        s = sum(input_rb)
        return [float(i)/s for i in input_rb]
    
    
    def predict(self, text):
        input_rb, input_bert = self.process_text(text)
        combined = np.append(input_rb, input_bert, axis=1)
        result = {0: 'A', 1: 'B'}
        return result[round(self.model.predict(combined)[0][0])]
    
    
    def process_text(self, text):
        doc = Document(Lang.RU, text)
        cna_graph_ru = CnaGraph(docs=doc, models=[self.ru_word2vec])
        compute_indices(doc, cna_graph_ru)
        input_rb = []
        input_bert = []
        for index in self.rb_indices_order:
            for key, value in doc.indices.items():
                if str(key) == index:
                    input_rb.append(value if value else 0)
                    break
            else:
                input_rb.append(0)
        xv = self.get_bert_embedding(text)
        input_bert.append(xv)
        input_rb = self.normalize_rb(input_rb)
        return [input_rb], input_bert
    

if __name__ == "__main__":
    r = RussianAvsB()
    text = "Она, как авторитетно утверждают мои родители и начальники, родилась раньше меня."
    print(r.predict(text))