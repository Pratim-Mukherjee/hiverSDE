"""Create a human-review golden set for final submission.

This script produces a reviewable CSV from the real data and a minimal set of
columns for a human to fill in. It does not pretend the labels are final.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data_prep import get_pairs
from src.intents import keyword_label

TARGET_TOTAL = 200
PER_INTENT_TARGET = TARGET_TOTAL // len(config.INTENTS)


def main() -> None:
    df = get_pairs()
    df["suggested_intent"] = df["customer_text"].apply(keyword_label)

    chunks = []
    rng_state = config.RANDOM_SEED
    for intent in config.INTENTS:
        bucket = df[df["suggested_intent"] == intent]
        n = min(PER_INTENT_TARGET, len(bucket))
        if n > 0:
            chunks.append(bucket.sample(n, random_state=rng_state))

    stratified = pd.concat(chunks) if chunks else pd.DataFrame(columns=df.columns)
    remaining_needed = TARGET_TOTAL - len(stratified)
    if remaining_needed > 0:
        leftover_pool = df.drop(stratified.index, errors="ignore")
        topup_n = min(remaining_needed, len(leftover_pool))
        topup = leftover_pool.sample(topup_n, random_state=rng_state + 1)
        stratified = pd.concat([stratified, topup])

    golden = stratified.sample(frac=1.0, random_state=rng_state + 2).reset_index(drop=True)
    golden["true_intent"] = ""
    golden["true_action"] = ""
    golden["review_notes"] = ""
    golden["reviewer"] = ""

    cols = [
        "thread_id",
        "customer_text",
        "brand_text",
        "suggested_intent",
        "true_intent",
        "true_action",
        "review_notes",
        "reviewer",
    ]

    out_path = config.EVAL_DIR / "golden_eval_reviewed.csv"
    golden[cols].to_csv(out_path, index=False)
    print(f"Wrote review template to {out_path}")
    print("Fill true_intent / true_action, then rename or copy the file to golden_eval.csv for final evaluation.")


if __name__ == "__main__":
    main()
