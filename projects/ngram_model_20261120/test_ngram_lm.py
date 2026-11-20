"""Tests for n-gram language model."""
from ngram_lm import NGramModel


def test_train():
    model = NGramModel(n=2)
    model.train([["hello", "world"]])
    assert len(model.ngram_counts) > 0


def test_log_probability():
    model = NGramModel(n=2, k=1.0)
    model.train([["a", "b"], ["a", "c"]])
    lp = model.log_probability("b", ("<s>",))
    assert lp < 0  # log prob should be negative


def test_perplexity():
    model = NGramModel(n=2, k=1.0)
    corpus = [["the", "cat"], ["the", "dog"]]
    model.train(corpus)
    pp = model.perplexity(corpus)
    assert pp > 0


def test_generate():
    model = NGramModel(n=2, k=0.1)
    model.train([["hello", "world"], ["hello", "there"]])
    generated = model.generate(max_len=10)
    assert isinstance(generated, list)
