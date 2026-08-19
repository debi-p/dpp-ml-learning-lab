from dataclasses import dataclass

import numpy as np

from backend.model_loader import load_packaged_model
from backend.text import normalize_for_display, tokenize


EPSILON = 1e-8


@dataclass
class MargRagEngine:
    model_id: str
    vocabulary: dict
    verses: list
    verse_embeddings: np.ndarray
    token_embeddings: np.ndarray
    projection: np.ndarray
    max_length: int

    @classmethod
    def load(cls, model_dir):
        model = load_packaged_model(model_dir)
        return cls(
            model_id=model.model_id,
            vocabulary=model.vocabulary,
            verses=model.verses,
            verse_embeddings=model.verse_embeddings,
            token_embeddings=model.token_embeddings,
            projection=model.projection,
            max_length=model.max_length,
        )

    def ask(self, question, top_k=3, intent_model=None):
        intent_result = intent_model.predict(question) if intent_model else None
        intent_result = override_core_gita_intent(question, intent_result)
        intent = intent_result["intent"] if intent_result else None
        verse_preferences = []
        if intent_result and intent_result.get("source") == "core_gita_rule" and intent_result.get("verse_preferences"):
            verse_preferences = intent_result.get("verse_preferences", [])
        elif intent_model and getattr(intent_model, "verse_preferences", None) and intent:
            verse_preferences = intent_model.verse_preferences.get(intent, [])
        elif intent_result:
            verse_preferences = intent_result.get("verse_preferences", [])
        results, boost_source = self.search(
            expand_query_for_intent(question, intent),
            top_k=top_k,
            intent=intent,
            verse_preferences=verse_preferences,
        )
        if intent_result is not None:
            intent_result["boost_source"] = boost_source
        return self.build_answer(question, results, intent_result=intent_result)

    def search(self, question, top_k=3, intent=None, verse_preferences=None):
        query = self.embed_text(question)
        scores = cosine_scores(query, self.verse_embeddings)
        scores, boost_source = boost_scores_for_intent(scores, self.verses, intent, verse_preferences=verse_preferences)
        order = scores.argsort()[::-1][:top_k]
        return [
            {
                "verse": self.verses[int(index)],
                "score": float(scores[int(index)]),
            }
            for index in order
        ], boost_source

    def embed_text(self, text):
        ids = encode_text(text, self.vocabulary, max_length=self.max_length)
        active = [token_id for token_id in ids if token_id != 0]
        if not active:
            pooled = np.zeros((self.token_embeddings.shape[1],), dtype=np.float32)
        else:
            pooled = self.token_embeddings[np.asarray(active, dtype=np.int64)].mean(axis=0)
        return l2_normalize(pooled @ self.projection).astype(np.float32)

    def build_answer(self, question, results, intent_result=None):
        if not results:
            return {"model_id": self.model_id, "question": question, "answer": "I could not find a relevant Gita passage.", "sources": []}

        sources = [format_source(result) for result in results]
        primary = sources[0]
        intent = intent_result["intent"] if intent_result else None
        guidance = topic_guidance(question, intent=intent)
        answer = (
            f"{primary['reference']} says: {primary['translation']} "
            f"{guidance}"
        )
        return {
            "model_id": self.model_id,
            "question": question,
            "intent": intent,
            "intent_confidence": intent_result["confidence"] if intent_result else None,
            "boost_source": intent_result.get("boost_source") if intent_result else "none",
            "answer": answer,
            "sources": sources,
        }


def encode_text(text, vocabulary, max_length):
    unk_id = vocabulary.get("<UNK>", 1)
    pad_id = vocabulary.get("<PAD>", 0)
    ids = [vocabulary.get(token, unk_id) for token in tokenize(text)]
    ids = ids[:max_length]
    if len(ids) < max_length:
        ids.extend([pad_id] * (max_length - len(ids)))
    return ids


def l2_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm < EPSILON:
        return vector
    return vector / norm


def cosine_scores(query_vector, matrix):
    query = l2_normalize(query_vector)
    matrix_norms = np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), EPSILON)
    return (matrix / matrix_norms) @ query


def truncate_text(text, max_chars=300):
    cleaned = normalize_for_display(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:") + "..."


def format_source(result):
    verse = result["verse"]
    return {
        "chapter": verse.get("chapter", ""),
        "verse": verse.get("verse", ""),
        "reference": f"Chapter {verse.get('chapter', '')}, Verse {verse.get('verse', '')}",
        "score": round(result["score"], 6),
        "translation": truncate_text(verse.get("translation", ""), max_chars=300),
        "commentary": truncate_text(verse.get("commentary", ""), max_chars=360),
        "tags": verse.get("tags", ""),
    }


def expand_query_for_intent(question, intent):
    hints = {
        "krishna_identity": "Krishna Supreme Lord source of all worlds truth cause of all causes everything rests upon Me",
        "devotion_worship": "worship devotion surrender Krishna Supreme Lord remember Me offer obeisances bhakti",
        "atma_paramatma": "atma soul paramatma supersoul body eternal heart overseer supreme soul",
        "self_identity": "soul self body eternal never dies changes bodies",
        "ego": "ego pride humility false ego self control service",
        "team_leadership": "duty leadership responsibility work qualities courage determination",
        "money_work": "work duty action fruits gain attachment honest effort",
        "anger": "anger desire senses control mind",
        "peace_stress": "peace mind attachment steady calm",
        "duty_decision": "duty action responsibility prescribed duty",
        "fear": "fear courage duty mind",
        "discipline": "practice mind control discipline steady",
        "desire_attachment": "desire attachment senses objects control",
        "success_failure": "success failure results fruits action",
        "relationship": "friend enemy equal vision compassion",
        "purpose": "purpose duty service action devotion",
    }
    return f"{question} {hints.get(intent, '')}".strip()


def boost_scores_for_intent(scores, verses, intent, verse_preferences=None):
    boosted = scores.copy()
    ranked_preferred = verse_preferences_to_ranked_keys(verse_preferences or [])
    preferred = set(ranked_preferred)
    boost_source = "intent_verse_preferences" if ranked_preferred else "hardcoded_fallback"
    if not preferred:
        preferred = {
        "krishna_identity": {("10", "8"), ("7", "7"), ("7", "19")},
        "devotion_worship": {("9", "22"), ("9", "34"), ("10", "8"), ("18", "66")},
        "atma_paramatma": {("2", "20"), ("13", "23"), ("15", "15")},
        "self_identity": {("2", "13"), ("2", "20"), ("2", "22")},
        "anger": {("2", "62"), ("2", "63"), ("5", "23")},
        "money_work": {("2", "47"), ("3", "19"), ("3", "30")},
        "team_leadership": {("18", "43"), ("3", "21"), ("2", "31")},
        "ego": {("3", "27"), ("16", "4"), ("18", "58")},
        "peace_stress": {("2", "70"), ("6", "26"), ("18", "66")},
        }.get(intent, set())
    if not preferred:
        return boosted, "none"
    boost = 0.8 if intent in {"krishna_identity", "devotion_worship", "atma_paramatma", "self_identity"} else 0.75
    for index, verse in enumerate(verses):
        key = (str(verse.get("chapter", "")), str(verse.get("verse", "")))
        if key in preferred:
            rank = ranked_preferred.get(key, 0)
            boosted[index] += boost if not ranked_preferred else boost * (1.0 + 0.5 / (rank + 1))
    return boosted, boost_source


def verse_preferences_to_ranked_keys(references):
    keys = {}
    for rank, reference in enumerate(references):
        parts = str(reference).strip().split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            keys.setdefault((str(int(parts[0])), str(int(parts[1]))), rank)
    return keys


def override_core_gita_intent(question, intent_result):
    text = question.lower()
    overrides = [
        (
            "krishna_identity",
            ["who is krishna", "what is krishna", "krishna who", "is krishna god", "bhagwan krishna"],
            0.99,
            ["10.8", "7.7", "7.19"],
        ),
        (
            "devotion_worship",
            ["worship", "pray", "prayer", "devotion", "devotional", "bhakti", "surrender", "krishna", "lord"],
            0.99,
            ["9.22", "9.34", "18.66", "10.8"],
        ),
        (
            "atma_paramatma",
            ["atma", "paramatma", "parmatma", "soul", "supersoul", "super soul"],
            0.99,
            ["13.23", "2.20", "15.15"],
        ),
        (
            "ego",
            ["ego", "pride", "arrogance", "arrogant", "humility"],
            0.99,
            ["3.27", "16.4", "18.58"],
        ),
    ]
    for intent, keywords, confidence, preferences in overrides:
        if any(keyword in text for keyword in keywords):
            return {
                "intent": intent,
                "confidence": confidence,
                "probabilities": {intent: confidence},
                "verse_preferences": preferences,
                "source": "core_gita_rule",
            }
    if any(phrase in text for phrase in ["am i a god", "am i god", "who am i", "what am i"]):
        return {
            "intent": "self_identity",
            "confidence": 0.99,
            "probabilities": {"self_identity": 0.99},
            "verse_preferences": ["2.13", "2.20", "2.22"],
            "source": "core_gita_rule",
        }
    return intent_result


def topic_guidance(question, intent=None):
    text = question.lower()
    if intent == "krishna_identity":
        return (
            "The Gita presents Krishna as the Supreme Lord, the source and support of everything. "
            "In daily life, this points to remembering the divine center behind action, worship, knowledge, and duty."
        )
    if intent == "devotion_worship":
        return (
            "For worship, the practical lesson is to direct devotion toward the Supreme Lord with sincerity, remembrance, and surrender. "
            "In daily life, this means making your work, choices, and mind less ego-centered and more connected to service."
        )
    if intent == "atma_paramatma":
        return (
            "Atma means the individual soul: eternal, conscious, and different from the temporary body. "
            "Paramatma means the Supersoul, the Lord present in the heart as witness, guide, and permitter."
        )
    if intent == "self_identity":
        return (
            "For identity, the practical lesson is that you are not the body, role, job, or emotion you are experiencing. "
            "Begin by observing yourself clearly, then act from the steadier self rather than from fear or confusion."
        )
    if intent == "ego":
        return (
            "For ego, the practical lesson is to reduce the need to prove yourself. "
            "Act sincerely, listen more, accept correction, and remember that ability is meant for service, not superiority."
        )
    if intent == "team_leadership":
        return (
            "For team and leadership, the practical lesson is to lead through responsibility, steadiness, and example. "
            "Guide people without ego, be clear about duty, and protect the team from confusion."
        )
    if intent == "money_work":
        return (
            "For earning and work, the practical lesson is to do honest duty without becoming greedy or ego-driven. "
            "Earn through right effort, use money responsibly, and do not let income become your identity."
        )
    if intent == "anger":
        return (
            "For anger, the practical lesson is to pause before reacting. "
            "Do not feed the first impulse. Step back, steady the senses, and then act from clarity."
        )
    if intent == "peace_stress":
        return (
            "For peace of mind, the practical lesson is to reduce attachment to outcomes. "
            "Do your part sincerely, then let the result come without constantly fighting it inside."
        )
    if intent == "duty_decision":
        return (
            "For duty, the practical lesson is to choose the action that is honest, useful, and aligned with your responsibility. "
            "Do not avoid the right action only because it is uncomfortable."
        )
    if intent == "fear":
        return (
            "For fear, the practical lesson is to bring the mind back to right action. "
            "Do the next truthful step instead of letting imagination control the whole situation."
        )
    if intent == "discipline":
        return (
            "For discipline, the practical lesson is steady practice. "
            "Do a small right action repeatedly, especially when comfort pulls you away."
        )
    if intent == "desire_attachment":
        return (
            "For desire and attachment, the practical lesson is to notice what is controlling the mind. "
            "Use self-control to choose what is good, not only what is immediately pleasant."
        )
    if intent == "success_failure":
        return (
            "For success and failure, the practical lesson is to focus on sincere effort rather than becoming shaken by results. "
            "Results matter, but they should not own your inner state."
        )
    if intent == "relationship":
        return (
            "For relationships, the practical lesson is to act with honesty, patience, and less ego. "
            "Try to understand before reacting, and choose words that reduce harm."
        )
    if intent == "purpose":
        return (
            "For purpose, the practical lesson is to connect your actions with duty, service, and inner growth. "
            "A meaningful life is built through sincere action, not only through personal gain."
        )
    if any(word in text for word in ["anger", "angry", "control anger", "temper"]):
        return (
            "For anger, the practical lesson is to pause before reacting. "
            "Do not feed the first impulse. Step back, steady the senses, and then act from clarity."
        )
    if any(word in text for word in ["earn", "money", "salary", "job", "career", "business", "work"]):
        return (
            "For earning and work, the practical lesson is to do honest duty without becoming greedy or ego-driven. "
            "Earn through right effort, use money responsibly, and do not let income become your identity."
        )
    if any(word in text for word in ["peace", "stress", "anxiety", "calm", "worry"]):
        return (
            "For peace of mind, the practical lesson is to reduce attachment to outcomes. "
            "Do your part sincerely, then let the result come without constantly fighting it inside."
        )
    if any(word in text for word in ["duty", "responsibility", "right thing", "decision", "confused"]):
        return (
            "For duty, the practical lesson is to choose the action that is honest, useful, and aligned with your responsibility. "
            "Do not avoid the right action only because it is uncomfortable."
        )
    return (
        "The practical lesson is to act with steadiness, self-control, and sincerity. "
        "Use the verse as a mirror for the situation, then choose the action that reduces ego and increases clarity."
    )
