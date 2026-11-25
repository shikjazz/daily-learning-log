"""Tests for text normalizer."""
from normalizer import (
    TextNormalizer, normalize_unicode, normalize_whitespace,
    normalize_quotes, remove_accents
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_quotes():
    result = normalize_quotes("\u201cHello\u201d")
    assert result == \'"Hello"\'


def test_remove_accents():
    assert remove_accents("café") == "cafe"
    assert remove_accents("naïve") == "naive"


def test_normalizer_pipeline():
    norm = TextNormalizer()
    result = norm.normalize("  Hello   World  ")
    assert result == "Hello World"


def test_normalizer_batch():
    norm = TextNormalizer()
    results = norm.normalize_batch(["  a  ", "  b  "])
    assert results == ["a", "b"]
