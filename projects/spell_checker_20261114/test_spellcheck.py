"""Tests for spell checker."""
from spellcheck import SpellChecker, levenshtein_distance


def test_levenshtein_same():
    assert levenshtein_distance("hello", "hello") == 0


def test_levenshtein_one_edit():
    assert levenshtein_distance("cat", "bat") == 1
    assert levenshtein_distance("cat", "cats") == 1


def test_is_correct():
    checker = SpellChecker()
    assert checker.is_correct("the")
    assert not checker.is_correct("xyz123")


def test_suggest():
    checker = SpellChecker()
    suggestions = checker.suggest("teh")
    words = [s[0] for s in suggestions]
    assert "the" in words


def test_correct():
    checker = SpellChecker()
    assert checker.correct("teh") == "the"
