"""
Evaluation harness.

Runs: TrivialBaseline, SimpleBaseline, and SupportAgent against the
hand-labelled golden set (eval/golden_eval.csv), and reports:
  - Intent classification: accuracy, macro-F1 (per system)
  - Escalation decision: precision/recall/F1 for ESCALATE_TO_HUMAN as the
    positive class (per system) -- this is a safety-relevant metric so we
    report both directions (missed escalations AND over-escalation).
  - Reply quality: mean judge score per axis, using judge.judge_reply
    (LLM if ANTHROPIC_API_KEY set, else heuristic -- clearly labeled in output).

Usage:
    python -m src.evaluate
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src import config
from src.data_prep import get_pairs
from src.intents import IntentClassifier
from src.retrieval import ReplyRetriever
from src.agent import SupportAgent
from src.baselines import TrivialBaseline, SimpleBaseline
from src.judge import judge_reply, llm_client


def load_golden():
    reviewed_path = config.EVAL_DIR / "golden_eval_reviewed.csv"
    legacy_path = config.EVAL_DIR / "golden_eval.csv"
    preferred = reviewed_path if reviewed_path.exists() else legacy_path

    if not preferred.exists():
        raise SystemExit(
            "No golden set found. Run `python -m src.cli golden` to generate a real-data candidate set, "
            "or `python -m src.cli review` to create a human-reviewed template."
        )

    if reviewed_path.exists():
        print(f"[evaluate] Using human-reviewed golden set: {reviewed_path}")
    elif legacy_path.exists():
        print(f"[evaluate] WARNING: using legacy golden_eval.csv; this is not submission-ready without a human-review pass.")

    df = pd.read_csv(preferred)
    if df["thread_id"].astype(str).str.lower().str.startswith("synth-").any():
        raise SystemExit(
            "Synthetic golden rows detected. Regenerate the gold file from the real TWCS data only."
        )
    missing = df["true_intent"].isna() | (df["true_intent"] == "")
    if missing.any():
        raise SystemExit(f"{missing.sum()} rows in {preferred.name} are missing true_intent labels.")
    return df


def build_agent_and_baselines():
    train_pairs = get_pairs()
    classifier = IntentClassifier().fit(train_pairs["customer_text"].tolist())
    retriever = ReplyRetriever(
        train_pairs["customer_text"].tolist(),
        train_pairs["brand_text"].tolist(),
        corpus_intents=[classifier.predict_one(t).intent for t in train_pairs["customer_text"]],
    )
    agent = SupportAgent(classifier, retriever, use_llm_polish=False)
    trivial = TrivialBaseline(train_pairs["customer_text"].tolist())
    simple = SimpleBaseline(retriever)
    return agent, trivial, simple


def eval_system(name, predict_fn, golden_df):
    intent_true, intent_pred = [], []
    action_true, action_pred = [], []
    judge_scores = []

    for _, row in golden_df.iterrows():
        out = predict_fn(row["customer_text"])
        intent_true.append(row["true_intent"])
        intent_pred.append(out["intent"])
        action_true.append(row["true_action"])
        action_pred.append(out["action"])

        evidence_text = ""
        if out.get("evidence"):
            evidence_text = out["evidence"][0].get("historical_brand_text", "")
        js = judge_reply(
            row["customer_text"], out["intent"], out["draft_reply"],
            evidence_text, grounded=(out["action"] == "AUTO_HANDLE"),
        )
        judge_scores.append(js.overall())

    intent_acc = accuracy_score(intent_true, intent_pred)
    intent_f1 = f1_score(intent_true, intent_pred, average="macro", zero_division=0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        action_true, action_pred, labels=["ESCALATE_TO_HUMAN"], average="macro", zero_division=0
    )

    return {
        "system": name,
        "intent_accuracy": round(intent_acc, 3),
        "intent_macro_f1": round(intent_f1, 3),
        "escalation_precision": round(prec, 3),
        "escalation_recall": round(rec, 3),
        "escalation_f1": round(f1, 3),
        "mean_judge_score": round(sum(judge_scores) / len(judge_scores), 3),
        "judge_source": "llm" if llm_client.is_available() else "heuristic",
    }


def main():
    golden_df = load_golden()
    agent, trivial, simple = build_agent_and_baselines()

    results = []
    results.append(eval_system("trivial_baseline", trivial.handle, golden_df))
    results.append(eval_system("simple_baseline", simple.handle, golden_df))
    results.append(eval_system("support_agent", lambda t: agent.handle(t).to_dict(), golden_df))

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    out_path = config.EVAL_DIR / "results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
