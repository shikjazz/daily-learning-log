"""
Word Embedding Similarity using Co-occurrence Matrices.
A from-scratch implementation of distributional semantics.
"""
import math
from collections import Counter, defaultdict


def build_cooccurrence(corpus: list[list[str]], window: int = 2) -> dict:
    """Build a co-occurrence matrix from a tokenized corpus."""
    cooccurrence = defaultdict(Counter)
    for sentence in corpus:
        for i, word in enumerate(sentence):
            start = max(0, i - window)
            end = min(len(sentence), i + window + 1)
            for j in range(start, end):
                if i != j:
                    cooccurrence[word][sentence[j]] += 1
    return dict(cooccurrence)


def ppmi(cooccurrence: dict) -> dict:
    """Compute Positive Pointwise Mutual Information from co-occurrence."""
    # Total count
    total = sum(sum(ctx.values()) for ctx in cooccurrence.values())
    # Word frequencies
    word_freq = {}
    for word, contexts in cooccurrence.items():
        word_freq[word] = sum(contexts.values())

    ppmi_matrix = {}
    for word, contexts in cooccurrence.items():
        ppmi_matrix[word] = {}
        for ctx, count in contexts.items():
            p_xy = count / total
            p_x = word_freq[word] / total
            p_y = word_freq.get(ctx, 1) / total
            pmi = math.log2(p_xy / (p_x * p_y)) if p_x > 0 and p_y > 0 else 0
            ppmi_matrix[word][ctx] = max(0, pmi)
    return ppmi_matrix


def cosine_sim(vec1: dict, vec2: dict) -> float:
    """Cosine similarity between two sparse vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    if not common:
        return 0.0
    dot = sum(vec1[k] * vec2[k] for k in common)
    mag1 = math.sqrt(sum(v**2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v**2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def most_similar(word: str, ppmi_matrix: dict, top_k: int = 5) -> list:
    """Find the most similar words to a given word."""
    if word not in ppmi_matrix:
        return []
    similarities = []
    for other_word in ppmi_matrix:
        if other_word != word:
            sim = cosine_sim(ppmi_matrix[word], ppmi_matrix[other_word])
            similarities.append((other_word, sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


if __name__ == "__main__":
    corpus = [
        ["the", "king", "rules", "the", "kingdom"],
        ["the", "queen", "rules", "the", "kingdom"],
        ["the", "prince", "is", "young", "royalty"],
        ["the", "princess", "is", "young", "royalty"],
        ["a", "dog", "chases", "a", "cat"],
        ["a", "cat", "runs", "from", "a", "dog"],
    ]
    cooc = build_cooccurrence(corpus, window=2)
    ppmi_mat = ppmi(cooc)
    print("Most similar to 'king':", most_similar("king", ppmi_mat))
    print("Most similar to 'dog':", most_similar("dog", ppmi_mat))
