#!/usr/bin/env python3
"""
Daily LLM/NLP Project Generator
Automatically generates a new mini NLP/ML project each day and commits it.
Each project is a self-contained Python module with documentation and tests.
"""

import os
import json
import random
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================================
# PROJECT TEMPLATES — Each is a complete mini-project
# ============================================================================

PROJECTS = [
    {
        "id": "text_classifier",
        "name": "Text Classifier",
        "description": "A simple text classifier using TF-IDF and cosine similarity",
        "tags": ["NLP", "classification", "TF-IDF"],
        "files": {
            "classifier.py": '''"""
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
''',
            "test_classifier.py": '''"""Tests for the text classifier."""
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
'''
        }
    },
    {
        "id": "word_embeddings",
        "name": "Word Embedding Similarity",
        "description": "Compute word similarities using co-occurrence based embeddings",
        "tags": ["NLP", "embeddings", "word2vec"],
        "files": {
            "embeddings.py": '''"""
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
''',
            "test_embeddings.py": '''"""Tests for word embeddings."""
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
'''
        }
    },
    {
        "id": "language_detector",
        "name": "Language Detector",
        "description": "Detect the language of text using character n-gram profiles",
        "tags": ["NLP", "language-detection", "n-grams"],
        "files": {
            "detector.py": '''"""
Language Detector using Character N-gram Profiles.
Identifies text language without any external libraries.
"""
from collections import Counter


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Extract character n-grams from text."""
    text = text.lower().strip()
    padded = f"__{text}__"
    return [padded[i:i+n] for i in range(len(padded) - n + 1)]


def build_profile(text: str, n: int = 3, top_k: int = 300) -> dict[str, int]:
    """Build a ranked n-gram profile from text."""
    ngrams = char_ngrams(text, n)
    counts = Counter(ngrams)
    ranked = {ng: rank for rank, (ng, _) in enumerate(counts.most_common(top_k))}
    return ranked


def profile_distance(profile1: dict, profile2: dict, max_penalty: int = 1000) -> int:
    """Compute out-of-place distance between two profiles."""
    distance = 0
    for ngram, rank1 in profile1.items():
        if ngram in profile2:
            distance += abs(rank1 - profile2[ngram])
        else:
            distance += max_penalty
    return distance


class LanguageDetector:
    """Detect language using character n-gram profile comparison."""

    TRAINING_DATA = {
        "english": "the of and to a in is it that was for on are with as his they be at one have this from by hot but some what there we can out other were all your when up use word how said each she which do their time if will way about many then them would write like so these her long make thing see him two has look more day could go come did my sound no most number who over know water than call first may down side been now find head stand own page should country found answer school grow study still learn plant cover food sun four thought let keep eye never last door between city tree cross since hard start might story saw far sea draw left late run while press close night real life few stop open seem together next white children begin got walk example ease paper often always music those both mark book letter until mile river car feet care second group carry took rain eat room friend began idea fish mountain north once base hear horse cut sure watch color face wood main enough plain girl usual young ready above ever red list though feel talk bird soon body dog family direct pose leave song measure state product black short numeral class wind question happen complete ship area half rock order fire south problem piece told knew pass farm top whole king size heard best hour better true during hundred remember step early hold west ground interest reach fast five sing listen six table travel less morning ten simple several vowel toward war lay against pattern slow center love person money serve appear road map science rule govern pull cold notice voice fall power town fine certain fly unit lead cry dark machine note wait plan figure star box noun field rest correct able pound done beauty drive stood contain front teach week final gave green oh quick develop sleep warm free minute strong special mind behind clear tail produce fact street inch lot nothing course stay wheel full force blue object decide surface deep moon island foot yet busy test record boat common gold possible plane age dry wonder laugh thousand ago ran check game shape yes cool miss brought heat snow bed bring sit perhaps fill east weight language among",
        "german": "der die und in den von zu das mit sich des auf fuer ist im dem nicht ein eine als auch es an werden aus er hat dass sie nach wird bei einer um am sind noch wie einem ueber einen so zum war haben nur oder aber vor zur bis mehr durch man sein wurde sei keine weil ein ihr ihre ihren einem eines einer ihrem sein seine seinem seinen seiner wurde war wenn zwei drei vier fuenf sechs sieben acht neun zehn Jahr Jahre Stadt Land Menschen Welt Zeit Frau Mann Kind Kinder Haus Arbeit Leben Tag Woche Ende Teil Seite Stadt Land Staat Bild Frage Weg Beispiel Zahl Hand Auge",
        "french": "de la le et les des en un une du est que pour dans ce il pas ne se par sur au plus qui sont avec tout mais nous comme ou son fait aussi bien autre lui ces encore avant entre deux sans notre dire elle peut premier ceux point monde pays fait jour homme temps bien peu connaitre vie nombre leur politique savoir coup etre faire comme tres ville pays monde temps",
        "spanish": "de la que el en y los del se las por un para con no una su al lo como mas pero sus le ya ha era muy sin sobre ser tiene tambien me hasta desde gobierno ciudad pais tiempo dia mundo parte casa vida mujer hombre hijo trabajo momento forma noche lugar palabra cosa caso numero manera punto agua grupo ciudad estado ejemplo sistema parte clase persona momento punto nivel tipo forma",
        "hindi": "ke hai ki ka aur ko se mein ne par hain yah ek ho is nahi kiya hai hum un sab kuch bhi kya tha woh thi kar jab tak ya us apne liye hai lekin sabhi baar sakte unke apni ye unhen koi dusre",
    }

    def __init__(self, ngram_size: int = 3):
        self.ngram_size = ngram_size
        self.profiles = {}
        self._train()

    def _train(self):
        """Build profiles from training data."""
        for lang, text in self.TRAINING_DATA.items():
            self.profiles[lang] = build_profile(text, self.ngram_size)

    def detect(self, text: str) -> str:
        """Detect the language of the input text."""
        text_profile = build_profile(text, self.ngram_size)
        best_lang = None
        best_distance = float("inf")
        for lang, profile in self.profiles.items():
            dist = profile_distance(text_profile, profile)
            if dist < best_distance:
                best_distance = dist
                best_lang = lang
        return best_lang

    def detect_probabilities(self, text: str) -> dict[str, float]:
        """Return language detection probabilities."""
        text_profile = build_profile(text, self.ngram_size)
        distances = {}
        for lang, profile in self.profiles.items():
            distances[lang] = profile_distance(text_profile, profile)

        # Convert distances to probabilities (inverse distance)
        max_dist = max(distances.values()) + 1
        inv_distances = {k: max_dist - v for k, v in distances.items()}
        total = sum(inv_distances.values())
        return {k: v / total for k, v in inv_distances.items()}


if __name__ == "__main__":
    detector = LanguageDetector()
    tests = [
        "This is a simple English sentence about natural language processing",
        "Dies ist ein einfacher deutscher Satz ueber Sprachverarbeitung",
        "Cette phrase est ecrite en francais pour tester la detection",
        "Esta es una oracion en espanol para probar la deteccion",
    ]
    for text in tests:
        lang = detector.detect(text)
        probs = detector.detect_probabilities(text)
        print(f"Text: {text[:50]}...")
        print(f"  Detected: {lang}")
        print(f"  Probabilities: {probs}")
        print()
''',
            "test_detector.py": '''"""Tests for language detector."""
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
'''
        }
    },
    {
        "id": "text_normalizer",
        "name": "Text Normalizer",
        "description": "Unicode-aware text normalization for NLP preprocessing",
        "tags": ["NLP", "preprocessing", "unicode"],
        "files": {
            "normalizer.py": '''"""
Text Normalizer for NLP Preprocessing.
Handles unicode normalization, whitespace, case folding, and more.
"""
import re
import unicodedata


def normalize_unicode(text: str, form: str = "NFKC") -> str:
    """Apply Unicode normalization (NFC, NFKC, NFD, NFKD)."""
    return unicodedata.normalize(form, text)


def remove_accents(text: str) -> str:
    """Remove diacritical marks from text."""
    nfkd = unicodedata.normalize("NFD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace into single spaces and strip."""
    return re.sub(r"\\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    return "".join(
        c for c in text
        if unicodedata.category(c)[0] != "C" or c in "\\n\\t"
    )


def normalize_quotes(text: str) -> str:
    """Normalize various quote characters to standard ASCII quotes."""
    # Single quotes
    text = re.sub(r"[\\u2018\\u2019\\u201A\\u201B\\u2039\\u203A]", "'", text)
    # Double quotes
    text = re.sub(r"[\\u201C\\u201D\\u201E\\u201F\\u00AB\\u00BB]", '"', text)
    return text


def normalize_dashes(text: str) -> str:
    """Normalize various dash characters to standard hyphens or em-dashes."""
    # En-dash and em-dash to standard
    text = text.replace("\\u2013", "-")  # en-dash
    text = text.replace("\\u2014", "--")  # em-dash
    text = text.replace("\\u2015", "--")  # horizontal bar
    return text


def normalize_numbers(text: str) -> str:
    """Normalize number representations."""
    # Replace fullwidth digits
    for i in range(10):
        text = text.replace(chr(0xFF10 + i), str(i))
    return text


class TextNormalizer:
    """Configurable text normalizer pipeline."""

    def __init__(self, steps=None):
        self.default_steps = [
            ("unicode", normalize_unicode),
            ("control_chars", remove_control_chars),
            ("quotes", normalize_quotes),
            ("dashes", normalize_dashes),
            ("numbers", normalize_numbers),
            ("whitespace", normalize_whitespace),
        ]
        if steps:
            self.steps = [(name, fn) for name, fn in self.default_steps if name in steps]
        else:
            self.steps = self.default_steps

    def normalize(self, text: str) -> str:
        """Apply all normalization steps."""
        for name, fn in self.steps:
            text = fn(text)
        return text

    def normalize_batch(self, texts: list[str]) -> list[str]:
        """Normalize a batch of texts."""
        return [self.normalize(t) for t in texts]


if __name__ == "__main__":
    normalizer = TextNormalizer()
    test_texts = [
        "Hello\\u2014World",
        "\\u201cSmart quotes\\u201d are annoying",
        "  Extra   whitespace   everywhere  ",
        "Caf\\u00e9 na\\u00efve r\\u00e9sum\\u00e9",
    ]
    for text in test_texts:
        result = normalizer.normalize(text)
        print(f"  Input:  {repr(text)}")
        print(f"  Output: {repr(result)}")
        print()
''',
            "test_normalizer.py": '''"""Tests for text normalizer."""
from normalizer import (
    TextNormalizer, normalize_unicode, normalize_whitespace,
    normalize_quotes, remove_accents
)


def test_normalize_whitespace():
    assert normalize_whitespace("  hello   world  ") == "hello world"


def test_normalize_quotes():
    result = normalize_quotes("\\u201cHello\\u201d")
    assert result == \\'\"Hello\"\\'


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
'''
        }
    },
    {
        "id": "sentiment_lexicon",
        "name": "Lexicon-Based Sentiment",
        "description": "Rule-based sentiment analysis using opinion lexicons",
        "tags": ["NLP", "sentiment", "lexicon"],
        "files": {
            "sentiment.py": '''"""
Lexicon-Based Sentiment Analyzer.
Uses a built-in opinion lexicon for rule-based sentiment scoring.
"""


# Compact opinion lexicon (subset of common sentiment words)
POSITIVE_WORDS = {
    "good", "great", "excellent", "amazing", "wonderful", "fantastic",
    "awesome", "outstanding", "superb", "brilliant", "love", "happy",
    "beautiful", "perfect", "best", "nice", "enjoy", "like", "impressive",
    "remarkable", "pleasant", "delightful", "magnificent", "terrific",
    "marvelous", "splendid", "favorable", "positive", "superior", "fine",
    "glad", "pleased", "satisfied", "thankful", "grateful", "cheerful",
    "exciting", "fun", "incredible", "successful", "effective", "useful",
    "helpful", "valuable", "worthy", "recommend", "praise", "win", "gain",
}

NEGATIVE_WORDS = {
    "bad", "terrible", "horrible", "awful", "poor", "worst", "hate",
    "ugly", "disgusting", "dreadful", "disappointing", "sad", "angry",
    "annoying", "boring", "stupid", "useless", "waste", "fail", "wrong",
    "broken", "nasty", "miserable", "pathetic", "inferior", "lousy",
    "painful", "unfortunate", "unpleasant", "dislike", "frustrated",
    "confused", "worried", "anxious", "stressed", "exhausted", "tired",
    "weak", "difficult", "problem", "error", "damage", "loss", "fear",
    "regret", "blame", "complain", "reject", "deny", "refuse",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "nobody", "nothing",
                  "nowhere", "nor", "cannot", "can't", "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "shouldn't", "isn't",
                  "aren't", "wasn't", "weren't", "haven't", "hasn't"}

INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "incredibly": 2.0,
                "really": 1.3, "absolutely": 2.0, "totally": 1.5,
                "quite": 1.2, "rather": 1.1, "somewhat": 0.8, "slightly": 0.5}


def tokenize(text: str) -> list[str]:
    """Simple tokenizer: lowercase, split, strip punctuation."""
    import re
    return re.findall(r"[a-z']+", text.lower())


def analyze_sentiment(text: str) -> dict:
    """Analyze sentiment of text, returning score and label."""
    tokens = tokenize(text)
    score = 0.0
    pos_words = []
    neg_words = []
    negation_active = False
    intensifier = 1.0

    for i, token in enumerate(tokens):
        if token in NEGATION_WORDS:
            negation_active = True
            continue

        if token in INTENSIFIERS:
            intensifier = INTENSIFIERS[token]
            continue

        if token in POSITIVE_WORDS:
            if negation_active:
                score -= 1.0 * intensifier
                neg_words.append(f"not {token}")
            else:
                score += 1.0 * intensifier
                pos_words.append(token)
        elif token in NEGATIVE_WORDS:
            if negation_active:
                score += 1.0 * intensifier
                pos_words.append(f"not {token}")
            else:
                score -= 1.0 * intensifier
                neg_words.append(token)

        # Reset modifiers after use
        negation_active = False
        intensifier = 1.0

    # Normalize by text length
    n_tokens = max(len(tokens), 1)
    normalized_score = score / (n_tokens ** 0.5)

    if normalized_score > 0.1:
        label = "positive"
    elif normalized_score < -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "score": round(normalized_score, 4),
        "label": label,
        "positive_words": pos_words,
        "negative_words": neg_words,
        "raw_score": score,
    }


if __name__ == "__main__":
    texts = [
        "This movie is absolutely amazing and wonderful!",
        "The food was terrible and the service was awful.",
        "It was not bad, actually quite good.",
        "The weather is okay, nothing special.",
        "I really hate this extremely boring and useless product.",
    ]
    for text in texts:
        result = analyze_sentiment(text)
        print(f"Text: {text}")
        print(f"  Label: {result['label']} (score: {result['score']})")
        print(f"  Positive: {result['positive_words']}")
        print(f"  Negative: {result['negative_words']}")
        print()
''',
            "test_sentiment.py": '''"""Tests for sentiment analyzer."""
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
'''
        }
    },
    {
        "id": "ngram_model",
        "name": "N-gram Language Model",
        "description": "Statistical language model using n-gram probabilities with smoothing",
        "tags": ["NLP", "language-model", "n-grams", "statistics"],
        "files": {
            "ngram_lm.py": '''"""
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
''',
            "test_ngram_lm.py": '''"""Tests for n-gram language model."""
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
'''
        }
    },
    {
        "id": "spell_checker",
        "name": "Spell Checker",
        "description": "Edit-distance based spell checker with frequency-weighted suggestions",
        "tags": ["NLP", "spell-checking", "edit-distance"],
        "files": {
            "spellcheck.py": '''"""
Edit-Distance Spell Checker.
Uses Levenshtein distance and word frequency for suggestions.
"""
from collections import Counter


# Common English words with approximate frequencies
WORD_FREQS = Counter({
    "the": 69971, "of": 36411, "and": 28852, "to": 26149, "a": 23237,
    "in": 21341, "is": 10595, "that": 10504, "for": 9489, "it": 9462,
    "was": 9321, "on": 8159, "are": 7370, "be": 6883, "with": 6862,
    "as": 6576, "at": 5508, "this": 5375, "have": 5307, "from": 5164,
    "or": 4793, "an": 4536, "by": 4336, "not": 4252, "but": 3842,
    "what": 3482, "all": 3392, "were": 3280, "when": 3199, "we": 3136,
    "there": 3021, "can": 2898, "been": 2721, "has": 2636, "more": 2541,
    "if": 2447, "no": 2398, "out": 2356, "do": 2323, "so": 2255,
    "up": 2210, "about": 2103, "into": 1955, "than": 1949, "its": 1903,
    "time": 1853, "only": 1785, "could": 1739, "new": 1691, "them": 1690,
    "some": 1615, "would": 1559, "make": 1531, "like": 1497, "just": 1485,
    "over": 1445, "know": 1436, "year": 1429, "take": 1400, "people": 1383,
    "come": 1360, "these": 1353, "him": 1352, "way": 1334, "find": 1312,
    "long": 1297, "day": 1280, "did": 1263, "get": 1259, "my": 1232,
    "language": 800, "processing": 600, "natural": 550, "learning": 500,
    "machine": 490, "model": 480, "data": 470, "algorithm": 350,
    "neural": 340, "network": 330, "training": 320, "text": 310,
    "word": 300, "sentence": 290, "token": 280, "embedding": 270,
})


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]


class SpellChecker:
    """Spell checker using edit distance and word frequency."""

    def __init__(self, dictionary: Counter = None):
        self.dictionary = dictionary or WORD_FREQS

    def is_correct(self, word: str) -> bool:
        """Check if a word is in the dictionary."""
        return word.lower() in self.dictionary

    def suggest(self, word: str, max_distance: int = 2, top_k: int = 5) -> list:
        """Suggest corrections for a misspelled word."""
        word = word.lower()
        if self.is_correct(word):
            return [(word, 0, self.dictionary[word])]

        candidates = []
        for dict_word, freq in self.dictionary.items():
            dist = levenshtein_distance(word, dict_word)
            if dist <= max_distance:
                candidates.append((dict_word, dist, freq))

        # Sort by distance first, then by frequency (higher = better)
        candidates.sort(key=lambda x: (x[1], -x[2]))
        return candidates[:top_k]

    def correct(self, word: str) -> str:
        """Return the best correction for a word."""
        suggestions = self.suggest(word)
        return suggestions[0][0] if suggestions else word

    def correct_text(self, text: str) -> str:
        """Correct all words in a text."""
        words = text.split()
        corrected = [self.correct(w) for w in words]
        return " ".join(corrected)


if __name__ == "__main__":
    checker = SpellChecker()
    test_words = ["teh", "langauge", "procesing", "machin", "lerning"]
    for word in test_words:
        suggestions = checker.suggest(word)
        print(f"'{word}' -> {suggestions[:3]}")
    print()
    text = "teh machin lerning algorthm"
    print(f"Original: {text}")
    print(f"Corrected: {checker.correct_text(text)}")
''',
            "test_spellcheck.py": '''"""Tests for spell checker."""
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
'''
        }
    },
]


def get_project_for_date(date: datetime) -> dict:
    """Deterministically select a project based on the date."""
    # Use date as seed for deterministic selection
    seed = int(date.strftime("%Y%m%d"))
    idx = seed % len(PROJECTS)
    return PROJECTS[idx]


def get_variation_suffix(date: datetime) -> str:
    """Generate a date-based variation suffix."""
    return date.strftime("%Y%m%d")


def create_project_files(project: dict, date: datetime, base_dir: str = "."):
    """Create all files for a project in the given directory."""
    suffix = get_variation_suffix(date)
    project_dir = os.path.join(base_dir, f"projects/{project['id']}_{suffix}")
    os.makedirs(project_dir, exist_ok=True)

    # Write project files
    for filename, content in project["files"].items():
        filepath = os.path.join(project_dir, filename)
        with open(filepath, "w") as f:
            f.write(content)

    # Write README
    readme_content = f"""# {project['name']}

> Generated on {date.strftime('%Y-%m-%d')} as part of daily NLP/ML learning

## Description

{project['description']}

## Tags

{', '.join(f'`{tag}`' for tag in project['tags'])}

## Files

{chr(10).join(f'- `{f}` — {"Main implementation" if "test" not in f else "Unit tests"}' for f in project['files'])}

## Usage

```python
python {list(project['files'].keys())[0]}
```

## Run Tests

```bash
python -m pytest {' '.join(f for f in project['files'] if 'test' in f)} -v
```

## Author

Shikhar Sharma — M.Sc. Computational Linguistics, University of Stuttgart
"""
    with open(os.path.join(project_dir, "README.md"), "w") as f:
        f.write(readme_content)

    return project_dir


def update_index(base_dir: str = "."):
    """Update the projects index in README.md."""
    projects_dir = os.path.join(base_dir, "projects")
    if not os.path.exists(projects_dir):
        return

    entries = sorted(os.listdir(projects_dir))

    readme = f"""# Daily LLM/NLP Projects

> Automatically generated NLP and ML projects for continuous learning.
> New project added daily via GitHub Actions.

## Projects ({len(entries)} total)

| Date | Project | Tags |
|------|---------|------|
"""
    for entry in reversed(entries):  # newest first
        parts = entry.rsplit("_", 1)
        if len(parts) == 2:
            proj_id, date_str = parts
            # Find matching project
            proj = next((p for p in PROJECTS if p["id"] == proj_id), None)
            if proj:
                date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                tags = ", ".join(f"`{t}`" for t in proj["tags"])
                readme += f"| {date_formatted} | [{proj['name']}](projects/{entry}/) | {tags} |\n"

    readme += f"""
## About

This repository contains daily auto-generated NLP/ML mini-projects covering:
- Text classification and sentiment analysis
- Word embeddings and distributional semantics
- Language detection and identification
- Text normalization and preprocessing
- N-gram language models
- Spell checking and correction
- And more!

Each project is self-contained with implementation, tests, and documentation.

**Author:** Shikhar Sharma — M.Sc. Computational Linguistics, University of Stuttgart
"""

    with open(os.path.join(base_dir, "README.md"), "w") as f:
        f.write(readme)


if __name__ == "__main__":
    today = datetime.now()
    project = get_project_for_date(today)
    print(f"Today's project: {project['name']}")
    print(f"Description: {project['description']}")

    project_dir = create_project_files(project, today)
    update_index()
    print(f"Created in: {project_dir}")
