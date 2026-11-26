"""Tests for sentiment analyzer."""
from sentiment import analyze_sentiment, tokenize


def test_tokenize():
    tokens = tokenize("Hello, World!")
    assert "hello" in tokens
    assert "world" in tokens


def test_positive_sentiment():
    result = analyze_sentiment("This is amazing and wonderful")
    assert result["label"] == "positive"
    assert result["score"] > 0


def test_negative_sentiment():
    result = analyze_sentiment("This is terrible and awful")
    assert result["label"] == "negative"
    assert result["score"] < 0


def test_negation():
    result = analyze_sentiment("This is not bad")
    assert result["score"] > 0  # "not bad" = positive


def test_neutral():
    result = analyze_sentiment("The table is made of wood")
    assert result["label"] == "neutral"
