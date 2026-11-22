"""
Text Classifier using TF-IDF and Cosine Similarity.
A lightweight text classification approach without deep learning.
"""
import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercasing tokenizer."""
    return text.lower().split()


def compute_tf(tokens: list[str]) -> dict[str, float]:
    """Compute term frequency for a list of tokens."""
    counts = Counter(tokens)
    total = len(tokens)
    return {word: count / total for word, count in counts.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency across documents."""
    n_docs = len(documents)
    idf = {}
    all_words = set(word for doc in documents for word in doc)
    for word in all_words:
        doc_count = sum(1 for doc in documents if word in doc)
        idf[word] = math.log(n_docs / (1 + doc_count))
    return idf


def compute_tfidf(tf: dict[str, float], idf: dict[str, float]) -> dict[str, float]:
    """Compute TF-IDF scores."""
    return {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Compute cosine similarity between two sparse vectors."""
    common = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[w] * vec2[w] for w in common)
    mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
    mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


class TextClassifier:
    """Simple TF-IDF based text classifier."""

    def __init__(self):
        self.classes = {}
        self.idf = {}
        self.class_vectors = {}

    def fit(self, texts: list[str], labels: list[str]):
        """Train the classifier on labeled texts."""
        tokenized = [tokenize(t) for t in texts]
        self.idf = compute_idf(tokenized)

        # Group by class
        class_texts = {}
        for tokens, label in zip(tokenized, labels):
            class_texts.setdefault(label, []).extend(tokens)

        # Compute class TF-IDF vectors
        for label, tokens in class_texts.items():
            tf = compute_tf(tokens)
            self.class_vectors[label] = compute_tfidf(tf, self.idf)

    def predict(self, text: str) -> str:
        """Predict the class of a text."""
        tokens = tokenize(text)
        tf = compute_tf(tokens)
        query_vec = compute_tfidf(tf, self.idf)

        best_label = None
        best_score = -1
        for label, class_vec in self.class_vectors.items():
            score = cosine_similarity(query_vec, class_vec)
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def predict_proba(self, text: str) -> dict[str, float]:
        """Predict class probabilities for a text."""
        tokens = tokenize(text)
        tf = compute_tf(tokens)
        query_vec = compute_tfidf(tf, self.idf)

        scores = {}
        for label, class_vec in self.class_vectors.items():
            scores[label] = cosine_similarity(query_vec, class_vec)

        total = sum(scores.values())
        if total > 0:
            return {k: v / total for k, v in scores.items()}
        return scores


if __name__ == "__main__":
    # Demo
    train_texts = [
        "machine learning neural network deep learning",
        "neural network backpropagation gradient descent",
        "the cat sat on the mat",
        "dogs are loyal pets and companions",
        "stock market trading financial portfolio",
        "investment banking hedge fund returns",
    ]
    train_labels = ["tech", "tech", "animals", "animals", "finance", "finance"]

    clf = TextClassifier()
    clf.fit(train_texts, train_labels)

    test = "artificial intelligence and deep neural networks"
    print(f"Text: {test}")
    print(f"Predicted: {clf.predict(test)}")
    print(f"Probabilities: {clf.predict_proba(test)}")
