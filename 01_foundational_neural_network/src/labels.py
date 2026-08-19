WORK_KEYWORDS = {
    "meeting",
    "project",
    "report",
    "review",
    "deadline",
    "office",
    "client",
    "call",
    "schedule",
    "today",
    "tomorrow",
    "urgent",
}

PROMOTION_KEYWORDS = {
    "offer",
    "free",
    "prize",
    "win",
    "claim",
    "discount",
    "voucher",
    "deal",
    "ringtone",
    "callertune",
    "subscription",
}


def build_label(category, message):
    normalized_category = category.strip().lower()
    message_words = set(message.lower().split())

    if normalized_category == "spam":
        return "spam"
    if message_words & PROMOTION_KEYWORDS:
        return "promotion"
    if message_words & WORK_KEYWORDS:
        return "work"
    return "personal"
