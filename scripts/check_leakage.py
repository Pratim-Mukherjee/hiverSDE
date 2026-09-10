"""Check whether the golden evaluation set leaked into the training/retrieval corpus.

This project is real-data-only. If the same customer_text appears in both the
training corpus and the golden evaluation set, the retriever is effectively
being evaluated on memorized rows and the metric is inflated.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.data_prep import get_pairs


def main() -> None:
    train_df = get_pairs()
    golden_df = pd.read_csv(config.EVAL_DIR / "golden_eval.csv")

    train_texts = train_df["customer_text"].astype(str).str.strip().str.lower()
    golden_texts = golden_df["customer_text"].astype(str).str.strip().str.lower()

    overlap_mask = train_texts.isin(golden_texts)
    overlap_rows = train_df.loc[overlap_mask].copy()

    overlap_pct = (len(overlap_rows) / len(train_df)) * 100.0 if len(train_df) else 0.0
    golden_overlap_pct = (len(overlap_rows) / len(golden_df)) * 100.0 if len(golden_df) else 0.0

    print("=== leakage check ===")
    print(f"training rows: {len(train_df)}")
    print(f"golden rows:   {len(golden_df)}")
    print(f"overlap rows:  {len(overlap_rows)}")
    print(f"training overlap: {overlap_pct:.2f}%")
    print(f"golden overlap:   {golden_overlap_pct:.2f}%")

    if len(overlap_rows) == 0:
        print("RESULT: no direct customer-text leakage detected.")
        return

    print("sample overlap rows:")
    print(overlap_rows[["thread_id", "customer_text"]].head(10).to_string(index=False))
    print("RESULT: leakage detected. Remove these rows from training before evaluation.")


if __name__ == "__main__":
    main()
