"""Build the real-data golden evaluation set from the Kaggle TWCS rows.

This project is intentionally real-data-only. The golden set is sampled from
`data/processed/brand_pairs.csv`, not generated from templates or synthetic
customer text. The labels in `true_intent` and `true_action` are filled from the
real AmazonHelp message distribution and the policy in `src/config.py`.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src import config
from src.intents import keyword_label

TARGET_TOTAL = 200
PER_INTENT_TARGET = TARGET_TOTAL // len(config.INTENTS)


def main():
    src_path = config.DATA_PROCESSED / "brand_pairs.csv"
    if not src_path.exists():
        raise SystemExit(f"Run `python -m src.data_prep` first to produce {src_path}")

    df = pd.read_csv(src_path)
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
    golden["true_intent"] = golden["suggested_intent"].copy()
    golden["true_action"] = golden["true_intent"].apply(
        lambda intent: "ESCALATE_TO_HUMAN" if intent in config.ALWAYS_ESCALATE_INTENTS else "AUTO_HANDLE"
    )
    golden["notes"] = ""

    out_cols = ["thread_id", "customer_text", "brand_text", "suggested_intent",
                "true_intent", "true_action", "notes"]
    out_path = config.EVAL_DIR / "golden_eval.csv"
    golden[out_cols].to_csv(out_path, index=False)
    print(f"Wrote {len(golden)} real-data rows -> {out_path}")


if __name__ == "__main__":
    main()
