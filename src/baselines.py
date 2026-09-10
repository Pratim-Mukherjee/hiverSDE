"""
Two baselines the real agent must beat -- required by the assignment.

TRIVIAL baseline:
    - Intent: always predict the single most frequent intent in training data.
    - Reply: always send the same generic canned message.
    - Escalation: never escalate (always AUTO_HANDLE).
  This is intentionally almost useless -- it's the floor.

SIMPLE baseline:
    - Intent: keyword lexicon only (config.INTENT_KEYWORDS), no ML, no fallback logic.
    - Reply: nearest-neighbor retrieval (same retriever as the real agent) but with
      NO similarity threshold -- always returns the top-1 match's reply, however dissimilar.
    - Escalation: single flat rule -- escalate only on ESCALATION_TRIGGER_PHRASES,
      ignoring intent-based policy and ignoring confidence.
  This represents "the obvious first thing an engineer would ship" -- a keyword
  router with retrieval, no confidence-awareness, no policy-based safety net.
"""

from collections import Counter
from . import config
from .intents import keyword_label
from .retrieval import ReplyRetriever


class TrivialBaseline:
    CANNED_REPLY = "Thanks for reaching out! Please DM us more details and we'll take a look."

    def __init__(self, train_texts, train_labels=None):
        labels = train_labels or [keyword_label(t) for t in train_texts]
        self.majority_intent = Counter(labels).most_common(1)[0][0]

    def handle(self, customer_text: str):
        return {
            "intent": self.majority_intent,
            "draft_reply": self.CANNED_REPLY,
            "action": "AUTO_HANDLE",
            "reason": "Trivial baseline: always predicts majority intent, always auto-handles.",
        }


class SimpleBaseline:
    def __init__(self, retriever: ReplyRetriever):
        self.retriever = retriever

    def handle(self, customer_text: str):
        intent = keyword_label(customer_text)
        hits = self.retriever.retrieve(customer_text, k=1, intent_filter=None)
        draft = hits[0]["historical_brand_text"] if hits else "Please DM us for help."

        text_lower = customer_text.lower()
        escalate = any(p in text_lower for p in config.ESCALATION_TRIGGER_PHRASES)
        return {
            "intent": intent,
            "draft_reply": draft,
            "action": "ESCALATE_TO_HUMAN" if escalate else "AUTO_HANDLE",
            "reason": "Simple baseline: keyword intent + top-1 retrieval regardless of "
                      "similarity, escalates only on hard trigger phrases.",
        }
