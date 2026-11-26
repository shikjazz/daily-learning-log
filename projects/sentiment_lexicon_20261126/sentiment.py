"""
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
