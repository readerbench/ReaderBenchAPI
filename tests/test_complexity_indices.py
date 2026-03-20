"""
Tests for building complexity indices across all supported languages.

Run with:
    env/bin/python -m pytest tests/test_complexity_indices.py -v
"""

import pytest

from rb.core.lang import Lang
from pipeline.parallel import build_features


SAMPLE_TEXTS = {
    Lang.EN: "The quick brown fox jumps over the lazy dog. It was a sunny afternoon in the park.",
    Lang.FR: "Le renard brun rapide saute par-dessus le chien paresseux. C'était une belle journée ensoleillée.",
    Lang.RO: "Vulpea brună sare peste câinele leneș. Era o după-amiază însorită în parc.",
    Lang.RU: "Быстрая коричневая лиса прыгает через ленивую собаку. Был солнечный день в парке.",
    Lang.PT: "A raposa marrom rápida pula sobre o cão preguiçoso. Era uma tarde ensolarada no parque.",
}


class TestBuildFeaturesBasic:
    """Tests that build_features returns a valid result for each language."""

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_returns_dict(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang)
        assert isinstance(result, dict), f"Expected dict for {lang.name}, got {type(result)}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_returns_nonempty_dict(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang)
        assert len(result) > 0, f"Empty indices dict for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_values_are_numeric_or_none(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang)
        for key, value in result.items():
            assert value is None or isinstance(value, (int, float)), (
                f"Index '{key}' has non-numeric value {value!r} for {lang.name}"
            )

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_at_least_some_indices_nonzero(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang)
        nonzero = [v for v in result.values() if v is not None and v != 0.0]
        assert len(nonzero) > 0, f"All indices are zero or None for {lang.name}"


class TestBuildFeaturesAllElements:
    """Tests that build_features with all_elements=True returns structured output."""

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_returns_elements_and_indices_keys(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang, all_elements=True)
        assert "elements" in result, f"Missing 'elements' key for {lang.name}"
        assert "indices" in result, f"Missing 'indices' key for {lang.name}"

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_elements_and_indices_same_length(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang, all_elements=True)
        assert len(result["elements"]) == len(result["indices"]), (
            f"elements/indices length mismatch for {lang.name}"
        )

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_first_element_is_document(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang, all_elements=True)
        assert result["elements"][0]["id"] == "Doc", (
            f"First element should be 'Doc' for {lang.name}"
        )

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_element_entries_have_id_and_text(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang, all_elements=True)
        for elem in result["elements"]:
            assert "id" in elem and "text" in elem, (
                f"Element missing 'id' or 'text' for {lang.name}: {elem}"
            )

    @pytest.mark.parametrize("lang", list(SAMPLE_TEXTS.keys()))
    def test_document_indices_nonempty(self, lang):
        result = build_features(SAMPLE_TEXTS[lang], lang, all_elements=True)
        doc_indices = result["indices"][0]
        assert isinstance(doc_indices, dict) and len(doc_indices) > 0, (
            f"Document-level indices dict is empty for {lang.name}"
        )
