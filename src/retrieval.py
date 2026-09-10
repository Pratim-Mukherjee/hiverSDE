"""
Retrieval-grounded reply drafting.

For a new customer message, find the k most similar historical customer
messages (by TF-IDF cosine similarity, optionally filtered to the same
predicted intent), then draft a reply by adapting the brand's own historical
reply pattern for that cluster -- not by asking an LLM to invent one from
scratch. This is what makes the reply "grounded in how the brand has
historically resolved similar issues" rather than generically plausible.

No paid API required. If ANTHROPIC_API_KEY is set in the environment, an
optional LLM polishing pass (src/llm_client.py) can rewrite the retrieved
template in more natural language while preserving its factual content --
but the retrieval step, not the LLM, is the source of truth.
"""

import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config


PLACEHOLDER_PATTERNS = [
    (re.compile(r"#\d{5,}"), "#{ORDER_ID}"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "{EMAIL}"),
]
SIGNOFF_PATTERNS = [
    re.compile(r"\s*\^?[A-Z]{2,5}\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:\^?[A-Z]{2,5}|[A-Z]{2,5})\s*$", re.IGNORECASE),
    re.compile(r"\s*(?:\^?[A-Z]{2,5})\s*(?:[.?!,;:]+\s*)*$", re.IGNORECASE),
]
THREAD_MARKER_PATTERNS = [
    re.compile(r"\s*\b\d+/\d+\s*$"),
    re.compile(r"\s*[\u2013\u2014\-–—]+\s*$"),
    re.compile(r"\s*[.:;,!?]+\s*$"),
]


def _strip_signoff(text: str) -> str:
    """Remove trailing internal agent initials such as ^HN or ^JD from a brand reply."""
    out = str(text or "")
    for pattern in SIGNOFF_PATTERNS:
        out = pattern.sub("", out)
    return out.strip()


def _cleanup_orphan_text(text: str) -> str:
    out = str(text or "")
    out = out.replace("\r", " ").replace("\n", " ")
    out = re.sub(r"\s+", " ", out).strip()
    for pattern in THREAD_MARKER_PATTERNS:
        out = pattern.sub("", out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    out = re.sub(r"([.,!?;:]){2,}", r"\1", out)
    return out.strip()


def _templatize(reply_text: str) -> str:
    """Replace brand-reply specifics (order numbers etc.) with placeholders so
    retrieved replies read as reusable templates rather than someone else's
    literal case details leaking into a new customer's reply."""
    out = _strip_signoff(reply_text)
    out = _cleanup_orphan_text(out)
    for pattern, placeholder in PLACEHOLDER_PATTERNS:
        out = pattern.sub(placeholder, out)
    return out


class ReplyRetriever:
    def __init__(self, corpus_customer_texts, corpus_brand_texts, corpus_intents=None):
        self.customer_texts = list(corpus_customer_texts)
        self.brand_texts = list(corpus_brand_texts)
        self.intents = list(corpus_intents) if corpus_intents is not None else None
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
        self.matrix = self.vectorizer.fit_transform(self.customer_texts)

    def retrieve(self, query_text: str, k: int = 3, intent_filter: str = None):
        q_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(q_vec, self.matrix)[0]

        idx_pool = np.arange(len(self.customer_texts))
        if intent_filter and self.intents is not None:
            filtered = [i for i in idx_pool if self.intents[i] == intent_filter]
            if filtered:  # only restrict if the filter isn't empty
                idx_pool = np.array(filtered)

        pool_sims = sims[idx_pool]
        top_local = np.argsort(-pool_sims)[:k]
        top_idx = idx_pool[top_local]

        results = []
        for i in top_idx:
            results.append({
                "similarity": float(sims[i]),
                "historical_customer_text": self.customer_texts[i],
                "historical_brand_text": self.brand_texts[i],
            })
        return results

    def draft_reply(self, query_text: str, intent_filter: str = None, k: int = 3):
        hits = self.retrieve(query_text, k=k, intent_filter=intent_filter)
        threshold = config.INTENT_SPECIFIC_GROUNDED_THRESHOLDS.get(intent_filter, config.MIN_RETRIEVAL_SIMILARITY_FOR_GROUNDED_REPLY)
        if not hits or hits[0]["similarity"] < threshold:
            return {
                "draft": None,
                "grounded": False,
                "evidence": hits,
                "reason": "No sufficiently similar historical resolution found (max sim "
                          f"{hits[0]['similarity']:.2f} < threshold {threshold:.2f})." if hits else "No corpus matches.",
            }
        best = hits[0]
        draft = _templatize(best["historical_brand_text"])
        return {
            "draft": draft,
            "grounded": True,
            "evidence": hits,
            "reason": f"Adapted from a historical reply to a similar {intent_filter or 'general'} "
                      f"message (similarity={best['similarity']:.2f}; threshold={threshold:.2f}).",
        }
