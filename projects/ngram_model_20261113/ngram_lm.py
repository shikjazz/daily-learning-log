"""
N-gram Language Model with Laplace Smoothing.
A from-scratch implementation of statistical language modeling.
"""
import math
import random
from collections import Counter, defaultdict


class NGramModel:
    """N-gram language model with add-k smoothing."""

    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, n: int = 3, k: float = 1.0):
        self.n = n
        self.k = k  # smoothing parameter
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()

    def _pad_sentence(self, tokens: list[str]) -> list[str]:
        """Add BOS and EOS tokens."""
        return [self.BOS] * (self.n - 1) + tokens + [self.EOS]

    def _get_ngrams(self, tokens: list[str]) -> list[tuple]:
        """Extract n-grams from a padded token list."""
        padded = self._pad_sentence(tokens)
        return [tuple(padded[i:i+self.n]) for i in range(len(padded) - self.n + 1)]

    def train(self, sentences: list[list[str]]):
        """Train the model on tokenized sentences."""
        for sentence in sentences:
            self.vocabulary.update(sentence)
            for ngram in self._get_ngrams(sentence):
                self.ngram_counts[ngram] += 1
                context = ngram[:-1]
                self.context_counts[context] += 1

        self.vocabulary.add(self.EOS)
        self.vocab_size = len(self.vocabulary)

    def log_probability(self, word: str, context: tuple) -> float:
        """Compute smoothed log probability P(word | context)."""
        ngram = context + (word,)
        count = self.ngram_counts[ngram]
        context_count = self.context_counts[context]
        prob = (count + self.k) / (context_count + self.k * self.vocab_size)
        return math.log2(prob) if prob > 0 else float("-inf")

    def sentence_log_prob(self, tokens: list[str]) -> float:
        """Compute total log probability of a sentence."""
        padded = self._pad_sentence(tokens)
        total = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i-self.n+1:i])
            word = padded[i]
            total += self.log_probability(word, context)
        return total

    def perplexity(self, sentences: list[list[str]]) -> float:
        """Compute perplexity over a set of sentences."""
        total_log_prob = 0.0
        total_words = 0
        for sentence in sentences:
            total_log_prob += self.sentence_log_prob(sentence)
            total_words += len(sentence) + 1  # +1 for EOS
        avg_log_prob = total_log_prob / total_words
        return 2 ** (-avg_log_prob)

    def generate(self, max_len: int = 20, temperature: float = 1.0) -> list[str]:
        """Generate a sentence from the model."""
        context = [self.BOS] * (self.n - 1)
        result = []

        for _ in range(max_len):
            ctx = tuple(context[-(self.n-1):])
            candidates = []
            weights = []
            for word in self.vocabulary:
                log_p = self.log_probability(word, ctx)
                candidates.append(word)
                weights.append(2 ** (log_p / temperature))

            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            word = random.choices(candidates, weights=weights, k=1)[0]

            if word == self.EOS:
                break
            result.append(word)
            context.append(word)

        return result


if __name__ == "__main__":
    corpus = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
        ["the", "cat", "ate", "the", "fish"],
        ["the", "dog", "ate", "the", "bone"],
        ["a", "cat", "sat", "on", "a", "mat"],
    ]

    model = NGramModel(n=3, k=0.1)
    model.train(corpus)

    print("Perplexity:", model.perplexity(corpus))
    print("Generated:", " ".join(model.generate()))
    print("Log P('the cat sat'):", model.sentence_log_prob(["the", "cat", "sat"]))
