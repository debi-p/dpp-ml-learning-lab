import re
import unicodedata


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def remove_diacritics(text):
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in normalized if not unicodedata.combining(char))


def normalize_for_display(text):
    text = remove_diacritics(text or "")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("Krsna", "Krishna").replace("Krsnas", "Krishna's")
    text = text.replace("yoge", "yogi").replace("gosvame", "gosvami").replace("svame", "svami")
    return clean_text(text)


def normalize_for_search(text):
    text = normalize_for_display(text).lower()
    text = re.sub(r"[^a-z0-9']+", " ", text)
    return clean_text(text)


def tokenize(text):
    return re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+", normalize_for_search(text))
