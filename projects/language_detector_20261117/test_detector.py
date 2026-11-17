"""Tests for language detector."""
from detector import LanguageDetector, char_ngrams, build_profile


def test_char_ngrams():
    ngrams = char_ngrams("hi", 2)
    assert len(ngrams) > 0
    assert all(len(ng) == 2 for ng in ngrams)


def test_build_profile():
    profile = build_profile("hello world", 3)
    assert isinstance(profile, dict)
    assert len(profile) > 0


def test_detect_english():
    det = LanguageDetector()
    result = det.detect("The quick brown fox jumps over the lazy dog")
    assert result == "english"


def test_detect_german():
    det = LanguageDetector()
    result = det.detect("Der schnelle braune Fuchs springt ueber den faulen Hund")
    assert result == "german"


def test_detect_probabilities():
    det = LanguageDetector()
    probs = det.detect_probabilities("Hello world")
    assert sum(probs.values()) - 1.0 < 0.01
    assert all(v >= 0 for v in probs.values())
