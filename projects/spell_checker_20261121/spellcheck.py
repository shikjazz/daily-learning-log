"""
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
