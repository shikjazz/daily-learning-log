"""
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
    return re.sub(r"\s+", " ", text).strip()


def remove_control_chars(text: str) -> str:
    """Remove control characters except newlines and tabs."""
    return "".join(
        c for c in text
        if unicodedata.category(c)[0] != "C" or c in "\n\t"
    )


def normalize_quotes(text: str) -> str:
    """Normalize various quote characters to standard ASCII quotes."""
    # Single quotes
    text = re.sub(r"[\u2018\u2019\u201A\u201B\u2039\u203A]", "'", text)
    # Double quotes
    text = re.sub(r"[\u201C\u201D\u201E\u201F\u00AB\u00BB]", '"', text)
    return text


def normalize_dashes(text: str) -> str:
    """Normalize various dash characters to standard hyphens or em-dashes."""
    # En-dash and em-dash to standard
    text = text.replace("\u2013", "-")  # en-dash
    text = text.replace("\u2014", "--")  # em-dash
    text = text.replace("\u2015", "--")  # horizontal bar
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
        "Hello\u2014World",
        "\u201cSmart quotes\u201d are annoying",
        "  Extra   whitespace   everywhere  ",
        "Caf\u00e9 na\u00efve r\u00e9sum\u00e9",
    ]
    for text in test_texts:
        result = normalizer.normalize(text)
        print(f"  Input:  {repr(text)}")
        print(f"  Output: {repr(result)}")
        print()
