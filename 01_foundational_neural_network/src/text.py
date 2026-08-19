import re


def clean_text(message):
    lowered = message.lower()
    letters_numbers_spaces = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", letters_numbers_spaces).strip()


def tokenize(cleaned_message):
    if not cleaned_message:
        return []
    return cleaned_message.split()
