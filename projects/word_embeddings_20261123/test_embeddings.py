"""Tests for word embeddings."""
from embeddings import build_cooccurrence, ppmi, cosine_sim, most_similar


def test_build_cooccurrence():
    corpus = [["a", "b", "c"]]
    cooc = build_cooccurrence(corpus, window=1)
    assert "b" in cooc["a"] or "a" in cooc["b"]


def test_ppmi_non_negative():
    corpus = [["x", "y", "z"], ["x", "y"]]
    cooc = build_cooccurrence(corpus)
    p = ppmi(cooc)
    for word in p:
        for ctx in p[word]:
            assert p[word][ctx] >= 0


def test_cosine_sim_identical():
    v = {"a": 1.0, "b": 2.0}
    assert abs(cosine_sim(v, v) - 1.0) < 0.001


def test_cosine_sim_orthogonal():
    v1 = {"a": 1.0}
    v2 = {"b": 1.0}
    assert cosine_sim(v1, v2) == 0.0
