import pickle
import numpy as np
import spacy
from tf2crf import CRF, ModelWithCRFLossDSCLoss, ModelWithCRFLoss
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow.keras.layers import *
from copy import deepcopy


def load_pickle_dump(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def transform_pos(list_pos, pos_index_dict):
    return [pos_index_dict[p] for p in list_pos]


def add_delimiters_to_words(words):
    return ['\t' + word + '\n' for word in words]


def transform_data(input_texts, max_seq_length, input_token_index):
    bilstm_input_data = np.zeros(
        (len(input_texts), max_seq_length),
        dtype='int32')
    for i, input_text in enumerate(input_texts):
        for t, char in enumerate(input_text):
            bilstm_input_data[i, t] = input_token_index[char]
    return bilstm_input_data


def format_input_data(words, posses, max_seq_length, input_token_index, pos_index_dict):
    words = add_delimiters_to_words(words)
    posses = [p if p and p in pos_index_dict else "Z" for p in posses]
    posses = transform_pos(posses, pos_index_dict)
    posses = [np.array([x] * len(words[i]) + [0] * (max_seq_length - len(words[i]))) for i, x in enumerate(posses)]
    words = transform_data(words, max_seq_length, input_token_index)
    return np.array(words), np.array(posses)


def build_bilstm_model(vocab_size, max_seq_length, embedding_dim, pos_size, units, add_crf=False):
    em_input = Input((max_seq_length,), dtype=tf.int32, name="cuvinte")
    pos_input = Input((max_seq_length,), dtype=tf.int32, name="pos")
    ch_emb = Embedding(vocab_size, embedding_dim, mask_zero=True, input_length=max_seq_length)(em_input)
    pos_emb = Embedding(pos_size, 16, mask_zero=True, input_length=max_seq_length)(pos_input)
    cc = Concatenate()([ch_emb, pos_emb])
    bilstm = Bidirectional(LSTM(units, return_sequences=True))(cc)
    
    # cc = Concatenate()([bilstm, pos_emb])
    td1 = TimeDistributed(Dense(64, activation="relu"))(bilstm)
    if add_crf:
        crf = CRF(units=2)(td1)
        return Model(inputs=[em_input, pos_input], outputs=crf)
    else:
        td2 = TimeDistributed(Dense(1, activation="sigmoid"))(td1)
        rs = Lambda(reshape)(td2)
        return Model(inputs=[em_input, pos_input], outputs=[rs])


def decode(word, binary_array):
    array_index = 1
    decoded_word = ''
    for i, c in enumerate(word):
        decoded_word += c
        if binary_array[array_index] == 1:
            decoded_word += '-'
        array_index += 1
        if array_index >= len(binary_array):
            break
    return decoded_word


def syllabify(text, ro_model):
    max_seq_length = load_pickle_dump('api/syllables/max_sequence_length.dump')
    input_token_index = load_pickle_dump('api/syllables/input_token_index.dump')
    pos_index_dict = load_pickle_dump('api/syllables/pos_index_dict.dump')

    words = []
    posses = []
    initial_words = []
    doc = ro_model(text)
    for word in doc:
        words.append(word.text)
        initial_words.append(word.text)
        posses.append(word.tag_)
    

    vocab_size = len(input_token_index) + 1
    embedding_dim = 16
    pos_size = len(pos_index_dict) + 1
    units = 64

    words, posses = format_input_data(words, posses, max_seq_length, input_token_index, pos_index_dict)

    add_crf = True
    model = build_bilstm_model(vocab_size, max_seq_length, embedding_dim, pos_size, units, add_crf)

    model = ModelWithCRFLoss(model, sparse_target=True)
    model.compile(optimizer=Adam(learning_rate=0.001))
    model.built = True
    model.load_weights('api/syllables/nr_model1.hdf5')

    return [decode(initial_words[i], s) for i, s in enumerate(model.predict({"cuvinte": words, "pos": posses}))]



# if __name__ == '__main__':
#     ro_model = spacy.load('ro_core_news_lg')
#     print(syllabify("mama", ro_model))