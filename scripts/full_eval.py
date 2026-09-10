"""Full real-data evaluation with leakage filtering, per-intent diagnostics,
bootstrap confidence intervals, and a McNemar test versus the simple baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config
from src.agent import SupportAgent
from src.baselines import SimpleBaseline
from src.data_prep import get_pairs
from src.evaluate import load_golden
from src.intents import IntentClassifier
from src.retrieval import ReplyRetriever


LABELS = config.INTENTS


def remove_overlap(train_df: pd.DataFrame, golden_df: pd.DataFrame) -> pd.DataFrame:
    golden_texts = golden_df["customer_text"].astype(str).str.strip().str.lower()
    overlap_mask = train_df["customer_text"].astype(str).str.strip().str.lower().isin(golden_texts)
    leaked = train_df.loc[overlap_mask]
    clean = train_df.loc[~overlap_mask].copy()
    print(f"[full_eval] removed {len(leaked)} leaked rows from the training corpus")
    print(f"[full_eval] leak-free corpus size: {len(clean)}")
    return clean


def bootstrap_accuracy(y_true: list[str], y_pred: list[str], n_boot: int = 2000, seed: int = 7) -> tuple[float, float, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(accuracy_score(y_true[idx], y_pred[idx]))
    scores = np.asarray(scores)
    lo, hi = np.quantile(scores, [0.025, 0.975])
    return float(np.mean(scores)), float(lo), float(hi)


def mcnemar_pvalue(y_true: list[str], pred_a: list[str], pred_b: list[str]) -> tuple[float, int, int, int, int]:
    a = np.asarray(y_true)
    pa = np.asarray(pred_a)
    pb = np.asarray(pred_b)

    correct_a = pa == a
    correct_b = pb == a

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    d = int(np.sum(~correct_a & ~correct_b))
    if b + c == 0:
        return 1.0, b, c, d, int(np.sum(correct_a & correct_b))

    stat = ((abs(b - c) - 1.0) ** 2) / (b + c)
    p_value = float(chi2.sf(stat, df=1))
    return p_value, b, c, d, int(np.sum(correct_a & correct_b))


def compute_reference_similarity(pred_reply: str, ref_reply: str) -> float:
    if not pred_reply or not ref_reply:
        return 0.0
    return float(SequenceMatcher(None, str(pred_reply), str(ref_reply)).ratio())


def fmt_table(rows: list[dict]) -> None:
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> None:
    train_pairs = get_pairs()
    golden = load_golden()
    train_pairs = remove_overlap(train_pairs, golden)

    classifier = IntentClassifier().fit(train_pairs["customer_text"].tolist())
    retriever = ReplyRetriever(
        train_pairs["customer_text"].tolist(),
        train_pairs["brand_text"].tolist(),
        corpus_intents=[classifier.predict_one(t).intent for t in train_pairs["customer_text"]],
    )
    agent = SupportAgent(classifier, retriever, use_llm_polish=False)
    simple = SimpleBaseline(retriever)

    support_preds = []
    simple_preds = []
    support_actions = []
    simple_actions = []
    support_ref_scores = []
    simple_ref_scores = []
    true_intents = []
    true_actions = []
    support_confusion_true = []
    support_confusion_pred = []
    simple_confusion_true = []
    simple_confusion_pred = []

    for _, row in golden.iterrows():
        support_out = agent.handle(row["customer_text"]).to_dict()
        simple_out = simple.handle(row["customer_text"])

        support_preds.append(support_out["intent"])
        simple_preds.append(simple_out["intent"])
        support_actions.append(support_out["action"])
        simple_actions.append(simple_out["action"])

        true_intents.append(row["true_intent"])
        true_actions.append(row["true_action"])

        support_confusion_true.append(row["true_intent"])
        support_confusion_pred.append(support_out["intent"])
        simple_confusion_true.append(row["true_intent"])
        simple_confusion_pred.append(simple_out["intent"])

        support_ref_scores.append(compute_reference_similarity(support_out["draft_reply"], row["brand_text"]))
        simple_ref_scores.append(compute_reference_similarity(simple_out["draft_reply"], row["brand_text"]))

    support_acc = accuracy_score(true_intents, support_preds)
    simple_acc = accuracy_score(true_intents, simple_preds)

    print("=== intent accuracy ===")
    print(f"support_agent: {support_acc:.4f}")
    print(f"simple_baseline: {simple_acc:.4f}")

    support_ci = bootstrap_accuracy(true_intents, support_preds)
    simple_ci = bootstrap_accuracy(true_intents, simple_preds)
    print("=== bootstrap 95% CI (intent accuracy) ===")
    print(f"support_agent: mean={support_ci[0]:.4f}, 95% CI=({support_ci[1]:.4f}, {support_ci[2]:.4f})")
    print(f"simple_baseline: mean={simple_ci[0]:.4f}, 95% CI=({simple_ci[1]:.4f}, {simple_ci[2]:.4f})")

    p_val, b, c, d, both = mcnemar_pvalue(true_intents, support_preds, simple_preds)
    print("=== McNemar test vs simple_baseline ===")
    print(f"support_correct_only={b}, baseline_correct_only={c}, both_correct={both}, both_wrong={d}")
    print(f"p-value={p_val:.6g}")

    support_conf = confusion_matrix(support_confusion_true, support_confusion_pred, labels=LABELS)
    simple_conf = confusion_matrix(simple_confusion_true, simple_confusion_pred, labels=LABELS)

    print("=== confusion matrix (support_agent) ===")
    print("rows=true intent, cols=predicted intent")
    cm_df = pd.DataFrame(support_conf, index=LABELS, columns=LABELS)
    print(cm_df.to_string())

    print("=== confusion matrix (simple_baseline) ===")
    print("rows=true intent, cols=predicted intent")
    print(pd.DataFrame(simple_conf, index=LABELS, columns=LABELS).to_string())

    support_prec, support_rec, support_f1, _ = precision_recall_fscore_support(
        true_intents, support_preds, labels=LABELS, average=None, zero_division=0
    )
    simple_prec, simple_rec, simple_f1, _ = precision_recall_fscore_support(
        true_intents, simple_preds, labels=LABELS, average=None, zero_division=0
    )

    per_intent_rows = []
    for intent, sp, sr, sf, bp, br, bf in zip(LABELS, support_prec, support_rec, support_f1, simple_prec, simple_rec, simple_f1):
        per_intent_rows.append({
            "intent": intent,
            "support_precision": round(float(sp), 3),
            "support_recall": round(float(sr), 3),
            "support_f1": round(float(sf), 3),
            "simple_precision": round(float(bp), 3),
            "simple_recall": round(float(br), 3),
            "simple_f1": round(float(bf), 3),
        })
    print("=== per-intent metrics ===")
    fmt_table(per_intent_rows)

    action_true = np.asarray(true_actions)
    support_action_acc = accuracy_score(action_true, support_actions)
    simple_action_acc = accuracy_score(action_true, simple_actions)
    print("=== escalation/action accuracy ===")
    print(f"support_agent: {support_action_acc:.4f}")
    print(f"simple_baseline: {simple_action_acc:.4f}")

    print("=== reference-based reply quality (higher is better) ===")
    print(f"support_agent mean similarity to brand_text: {np.mean(support_ref_scores):.4f}")
    print(f"simple_baseline mean similarity to brand_text: {np.mean(simple_ref_scores):.4f}")


if __name__ == "__main__":
    main()
