"""Tests for the text classifier."""
from classifier import TextClassifier, tokenize, compute_tf, cosine_similarity


def test_tokenize():
    assert tokenize("Hello World") == ["hello", "world"]


def test_compute_tf():
    tf = compute_tf(["a", "b", "a"])
    assert abs(tf["a"] - 2/3) < 0.01
    assert abs(tf["b"] - 1/3) < 0.01


def test_cosine_similarity():
    v1 = {"a": 1.0, "b": 1.0}
    v2 = {"a": 1.0, "b": 1.0}
    assert abs(cosine_similarity(v1, v2) - 1.0) < 0.01


def test_classifier():
    clf = TextClassifier()
    clf.fit(
        ["cat dog pet", "stock market money", "cat purrs softly"],
        ["animals", "finance", "animals"]
    )
    assert clf.predict("little kitten pet") == "animals"


def test_predict_proba():
    clf = TextClassifier()
    clf.fit(["hello world", "goodbye moon"], ["greet", "farewell"])
    probs = clf.predict_proba("hello there")
    assert "greet" in probs
    assert sum(probs.values()) - 1.0 < 0.01 or sum(probs.values()) == 0
