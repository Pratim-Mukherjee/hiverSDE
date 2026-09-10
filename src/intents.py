"""
Intent classification.

Approach (deliberately NOT an LLM call, so the pipeline is free and fast):
  1. Weak-label every training pair using the keyword lexicon in config.py.
  2. Train a TF-IDF + Logistic Regression multiclass classifier on those
     weak labels.
  3. At inference, take the ML model's prediction, but if its top-class
     probability is low, fall back to the keyword rule (more conservative,
     fully explainable) and flag `used_fallback=True`.

Why not just use the keyword rules directly? They're high-precision but
low-recall/brittle to phrasing (e.g. "my package never came" won't match
literal keywords). The ML layer generalizes; the keyword layer is the
safety net when the ML layer is unsure. This hybrid is documented in
DECISION_LOG.md.
"""

from dataclasses import dataclass
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config


def keyword_label(text: str) -> str:
    t = text.lower()
    for intent, kws in config.INTENT_KEYWORDS.items():
        for kw in kws:
            if kw in t:
                return intent
    return "general_inquiry_or_other"


def weak_label_dataset(texts):
    return [keyword_label(t) for t in texts]


@dataclass
class IntentResult:
    intent: str
    confidence: float
    used_fallback: bool


class IntentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced")),
        ])
        self._fitted = False

    def fit(self, texts):
        labels = weak_label_dataset(texts)
        self.pipeline.fit(texts, labels)
        self._fitted = True
        return self

    def predict_one(self, text: str) -> IntentResult:
        if not self._fitted:
            raise RuntimeError("Call .fit() before predicting.")
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_idx = int(np.argmax(proba))
        top_intent, top_conf = classes[top_idx], float(proba[top_idx])

        if top_conf < config.MIN_CLASSIFIER_CONFIDENCE_FOR_AUTOHANDLE:
            kw_intent = keyword_label(text)
            if kw_intent != "general_inquiry_or_other":
                return IntentResult(kw_intent, top_conf, used_fallback=True)
        return IntentResult(top_intent, top_conf, used_fallback=False)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    def load(self, path):
        self.pipeline = joblib.load(path)
        self._fitted = True
        return self
