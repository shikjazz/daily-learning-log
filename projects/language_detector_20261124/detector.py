"""
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
