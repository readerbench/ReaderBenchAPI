"""
Tests for loading vector models (Transformers, Word2Vec, LSA, LDA) across all supported languages.

Run with:
    env/bin/python -m pytest tests/test_vector_models.py -v

Word2Vec/LSA/LDA models are downloaded automatically from the ReaderBench server on first use.
"""

import pytest
import numpy as np

from rb.core.lang import Lang
from rb.similarity.lda import LDA
from rb.similarity.lsa import LSA
from rb.similarity.transformers_encoder import TransformersEncoder
from rb.similarity.vector_model import VectorModelType
from rb.similarity.vector_model_factory import create_vector_model
from rb.similarity.word2vec import Word2Vec
from rb import Document


# Short representative texts per language for encoding tests
SAMPLE_TEXTS = {
    Lang.EN: "The quick brown fox jumps over the lazy dog.",
    Lang.FR: "Le renard brun rapide saute par-dessus le chien paresseux.",
    Lang.RO: "Vulpea brună sare peste câinele leneș.",
    Lang.RU: "Быстрая коричневая лиса прыгает через ленивую собаку.",
    Lang.PT: "A raposa marrom rápida pula sobre o cão preguiçoso.",
}

# Model names expected per language (mirrors TransformersEncoder.__init__)
EXPECTED_MODEL_NAMES = {
    Lang.EN: "sentence-transformers/all-distilroberta-v1",
    Lang.FR: "camembert-base",
    Lang.RO: "readerbench/RoBERT-base",
    Lang.RU: "DeepPavlov/rubert-base-cased",
    Lang.PT: "neuralmind/bert-base-portuguese-cased",
}


class TestTransformersEncoderLoading:
    """Tests that TransformersEncoder instantiates correctly for each language."""

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_instantiation(self, lang):
        encoder = TransformersEncoder(lang)
        assert encoder is not None
        assert encoder.lang == lang
        assert encoder.type == VectorModelType.TRANSFORMER
        assert encoder.size == 768

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_correct_model_loaded(self, lang):
        encoder = TransformersEncoder(lang)
        assert encoder.name == EXPECTED_MODEL_NAMES[lang]

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_tokenizer_loaded(self, lang):
        encoder = TransformersEncoder(lang)
        assert encoder.tokenizer is not None

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_bert_model_loaded(self, lang):
        encoder = TransformersEncoder(lang)
        assert encoder.bert is not None

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_bos_eos_tokens_set(self, lang):
        encoder = TransformersEncoder(lang)
        assert encoder.bos is not None
        assert encoder.eos is not None


class TestTransformersEncoderEncoding:
    """Tests that TransformersEncoder produces valid vectors when encoding documents."""

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_encode_document_produces_vector(self, lang):
        encoder = TransformersEncoder(lang)
        doc = Document(lang, SAMPLE_TEXTS[lang])
        encoder.encode(doc)
        assert encoder in doc.vectors, f"No document vector produced for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_encoded_vector_shape(self, lang):
        encoder = TransformersEncoder(lang)
        doc = Document(lang, SAMPLE_TEXTS[lang])
        encoder.encode(doc)
        vec = doc.vectors[encoder].values
        assert vec.shape == (768,), f"Expected shape (768,), got {vec.shape} for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_encoded_vector_is_finite(self, lang):
        encoder = TransformersEncoder(lang)
        doc = Document(lang, SAMPLE_TEXTS[lang])
        encoder.encode(doc)
        vec = doc.vectors[encoder].values
        assert np.all(np.isfinite(vec)), f"Vector contains non-finite values for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_encoded_vector_nonzero(self, lang):
        encoder = TransformersEncoder(lang)
        doc = Document(lang, SAMPLE_TEXTS[lang])
        encoder.encode(doc)
        vec = doc.vectors[encoder].values
        assert np.any(vec != 0), f"Vector is all zeros for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_word_vectors_produced(self, lang):
        encoder = TransformersEncoder(lang)
        doc = Document(lang, SAMPLE_TEXTS[lang])
        encoder.encode(doc)
        words_with_vectors = [
            word for block in doc.get_blocks()
            for sentence in block.get_sentences()
            for word in sentence.get_words()
            if encoder in word.vectors
        ]
        assert len(words_with_vectors) > 0, f"No word vectors produced for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_different_texts_produce_different_vectors(self, lang):
        encoder = TransformersEncoder(lang)
        text1 = SAMPLE_TEXTS[lang]
        # Reverse the sentence to get a meaningfully different text
        words = text1.split()
        text2 = " ".join(reversed(words))
        doc1 = Document(lang, text1)
        doc2 = Document(lang, text2)
        encoder.encode(doc1)
        encoder.encode(doc2)
        vec1 = doc1.vectors[encoder].values
        vec2 = doc2.vectors[encoder].values
        assert not np.allclose(vec1, vec2), f"Different texts produced identical vectors for {lang.name}"


class TestCreateVectorModelFactory:
    """Tests that create_vector_model correctly instantiates a TransformersEncoder."""

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_create_transformer_model(self, lang):
        model = create_vector_model(lang, VectorModelType.TRANSFORMER, corpus="")
        assert model is not None
        assert isinstance(model, TransformersEncoder)
        assert model.lang == lang

    @pytest.mark.parametrize("lang", list(EXPECTED_MODEL_NAMES.keys()))
    def test_create_transformer_model_caching(self, lang):
        """Second call with same params returns the cached instance."""
        model1 = create_vector_model(lang, VectorModelType.TRANSFORMER, corpus="")
        model2 = create_vector_model(lang, VectorModelType.TRANSFORMER, corpus="")
        assert model1 is model2


# ---------------------------------------------------------------------------
# Word2Vec tests
# ---------------------------------------------------------------------------

# (lang, corpus, a known word expected to be in the vocabulary)
WORD2VEC_CONFIGS = [
    (Lang.EN,  "coca",          "king"),
    (Lang.RO,  "readme",        "și"),
    (Lang.FR,  "le_monde",      "le"),
    (Lang.RU,  "rnc_wikipedia", "он"),
    (Lang.ES,  "jose_antonio",  "de"),
    (Lang.NL,  "wiki",          "de"),
    (Lang.DE,  "wiki",          "die"),
]

WORD2VEC_IDS = [f"{lang.name}-{corpus}" for lang, corpus, _ in WORD2VEC_CONFIGS]


class TestWord2VecLoading:
    """Tests that Word2Vec instantiates and loads correctly for each language."""

    @pytest.mark.parametrize("lang,corpus,_", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_instantiation(self, lang, corpus, _):
        model = Word2Vec(corpus, lang)
        assert model is not None
        assert model.lang == lang
        assert model.type == VectorModelType.WORD2VEC
        assert model.size > 0

    @pytest.mark.parametrize("lang,corpus,_", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_vectors_loaded(self, lang, corpus, _):
        model = Word2Vec(corpus, lang)
        assert len(model.vectors) > 0, f"No vectors loaded for {lang.name}/{corpus}"

    @pytest.mark.parametrize("lang,corpus,word", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_known_word_has_vector(self, lang, corpus, word):
        model = Word2Vec(corpus, lang)
        vec = model.get_vector(word)
        assert vec is not None, f"No vector for '{word}' in {lang.name}/{corpus}"

    @pytest.mark.parametrize("lang,corpus,word", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_vector_shape(self, lang, corpus, word):
        model = Word2Vec(corpus, lang)
        vec = model.get_vector(word)
        assert vec is not None
        assert vec.values.ndim == 1
        assert len(vec.values) == model.size

    @pytest.mark.parametrize("lang,corpus,_", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_vector_values_finite(self, lang, corpus, _):
        model = Word2Vec(corpus, lang)
        sample = list(model.vectors.values())[:100]
        for vec in sample:
            assert np.all(np.isfinite(vec.values)), f"Non-finite values in {lang.name}/{corpus}"

    @pytest.mark.parametrize("lang,corpus,_", WORD2VEC_CONFIGS, ids=WORD2VEC_IDS)
    def test_caching(self, lang, corpus, _):
        m1 = create_vector_model(lang, VectorModelType.WORD2VEC, corpus=corpus)
        m2 = create_vector_model(lang, VectorModelType.WORD2VEC, corpus=corpus)
        assert m1 is m2


class TestWord2VecSimilarity:
    """Tests Word2Vec similarity computation."""

    def test_self_similarity_is_one(self):
        model = Word2Vec("coca", Lang.EN)
        word = next((w for w in ("king", "the", "house") if w in model.vectors), None)
        if word is None:
            pytest.skip("No test word found in vocabulary")
        vec = model.get_vector(word)
        assert abs(model.similarity(vec, vec) - 1.0) < 1e-5

    def test_similar_words_score_higher_than_dissimilar(self):
        model = Word2Vec("coca", Lang.EN)
        for word_a, word_b, word_c in [("king", "queen", "banana"), ("dog", "cat", "mathematics")]:
            if not all(w in model.vectors for w in (word_a, word_b, word_c)):
                continue
            sim_related = model.similarity(model.get_vector(word_a), model.get_vector(word_b))
            sim_unrelated = model.similarity(model.get_vector(word_a), model.get_vector(word_c))
            assert sim_related > sim_unrelated, (
                f"Expected sim({word_a},{word_b}) > sim({word_a},{word_c}), "
                f"got {sim_related:.4f} vs {sim_unrelated:.4f}"
            )
            return
        pytest.skip("None of the test word pairs found in vocabulary")

    def test_different_words_not_identical(self):
        model = Word2Vec("coca", Lang.EN)
        words = [w for w in ("king", "queen", "dog", "cat", "house") if w in model.vectors]
        if len(words) < 2:
            pytest.skip("Not enough test words in vocabulary")
        vec_a = model.get_vector(words[0])
        vec_b = model.get_vector(words[1])
        assert not np.allclose(vec_a.values, vec_b.values)


# ---------------------------------------------------------------------------
# LSA tests  (only supported for EN/coca)
# ---------------------------------------------------------------------------

class TestLSALoading:
    """Tests that LSA instantiates and loads correctly (EN/coca)."""

    def test_instantiation(self):
        model = LSA("coca", Lang.EN)
        assert model is not None
        assert model.lang == Lang.EN
        assert model.type == VectorModelType.LSA
        assert model.size > 0

    def test_vectors_loaded(self):
        model = LSA("coca", Lang.EN)
        assert len(model.vectors) > 0

    def test_known_word_has_vector(self):
        model = LSA("coca", Lang.EN)
        word = next((w for w in ("king", "house", "dog") if w in model.vectors), None)
        assert word is not None, "No known word found in LSA vocabulary"
        assert model.get_vector(word) is not None

    def test_vector_values_finite(self):
        model = LSA("coca", Lang.EN)
        sample = list(model.vectors.values())[:100]
        for vec in sample:
            assert np.all(np.isfinite(vec.values))

    def test_self_similarity_is_one(self):
        model = LSA("coca", Lang.EN)
        word = next((w for w in ("king", "house", "dog") if w in model.vectors), None)
        if word is None:
            pytest.skip("No test word found in LSA vocabulary")
        vec = model.get_vector(word)
        assert abs(model.similarity(vec, vec) - 1.0) < 1e-5

    def test_caching(self):
        m1 = create_vector_model(Lang.EN, VectorModelType.LSA, corpus="coca")
        m2 = create_vector_model(Lang.EN, VectorModelType.LSA, corpus="coca")
        assert m1 is m2


# ---------------------------------------------------------------------------
# LDA tests  (only supported for EN/coca)
# ---------------------------------------------------------------------------

class TestLDALoading:
    """Tests that LDA instantiates and loads correctly (EN/coca)."""

    def test_instantiation(self):
        model = LDA("coca", Lang.EN)
        assert model is not None
        assert model.lang == Lang.EN
        assert model.type == VectorModelType.LDA
        assert model.size > 0

    def test_vectors_loaded(self):
        model = LDA("coca", Lang.EN)
        assert len(model.vectors) > 0

    def test_known_word_has_vector(self):
        model = LDA("coca", Lang.EN)
        word = next((w for w in ("king", "house", "dog") if w in model.vectors), None)
        assert word is not None, "No known word found in LDA vocabulary"
        assert model.get_vector(word) is not None

    def test_vector_values_finite(self):
        model = LDA("coca", Lang.EN)
        sample = list(model.vectors.values())[:100]
        for vec in sample:
            assert np.all(np.isfinite(vec.values))

    def test_self_similarity_is_one(self):
        model = LDA("coca", Lang.EN)
        word = next((w for w in ("king", "house", "dog") if w in model.vectors), None)
        if word is None:
            pytest.skip("No test word found in LDA vocabulary")
        vec = model.get_vector(word)
        assert abs(model.similarity(vec, vec) - 1.0) < 1e-5

    def test_caching(self):
        m1 = create_vector_model(Lang.EN, VectorModelType.LDA, corpus="coca")
        m2 = create_vector_model(Lang.EN, VectorModelType.LDA, corpus="coca")
        assert m1 is m2
