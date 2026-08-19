import re
import unicodedata


DISPLAY_REPLACEMENTS = {
    "Kanea": "Krishna",
    "Kaneas": "Krishna's",
    "Krsna": "Krishna",
    "Krsnas": "Krishna's",
    "Geta": "Gita",
    "Gita": "Gita",
    "jiana": "jnana",
    "yoge": "yogi",
    "yogi": "yogi",
    "gosvame": "gosvami",
    "gosvami": "gosvami",
    "svame": "svami",
    "svami": "svami",
    "jnana": "jnana",
}


SEARCH_REPLACEMENTS = {
    "krsna": "krishna",
    "gita": "gita",
    "yoge": "yogi",
    "gosvame": "gosvami",
    "svame": "svami",
}


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def remove_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_known_transliterations(text):
    replacements = {
        "Kåñëa": "Krishna",
        "Kåñëa's": "Krishna's",
        "kåñëa": "krishna",
        "kåñëa's": "krishna's",
        "Bhagavad-gétä": "Bhagavad Gita",
        "bhagavad-gétä": "bhagavad gita",
        "Gétä": "Gita",
        "gétä": "gita",
        "jïäna": "jnana",
        "Jïäna": "Jnana",
        "yogé": "yogi",
        "Yogé": "Yogi",
        "gosvämé": "gosvami",
        "Gosvämé": "Gosvami",
        "svämé": "svami",
        "Svämé": "Svami",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def normalize_special_words(text, replacements):
    def replace(match):
        word = match.group(0)
        lower = word.lower()
        replacement = replacements.get(lower)
        if replacement is None:
            return word
        if word[:1].isupper():
            return replacement[:1].upper() + replacement[1:]
        return replacement

    return re.sub(r"\b[A-Za-z']+\b", replace, text)


def normalize_for_display(text):
    text = normalize_known_transliterations(text or "")
    text = remove_diacritics(text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("―", "-").replace("–", "-").replace("—", "-")
    text = normalize_special_words(text, DISPLAY_REPLACEMENTS)
    return clean_text(text)


def normalize_for_search(text):
    text = normalize_for_display(text).lower()
    text = normalize_special_words(text, SEARCH_REPLACEMENTS)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return clean_text(text)


def tokenize(text):
    cleaned = normalize_for_search(text)
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+", cleaned)
