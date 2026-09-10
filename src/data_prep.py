"""
Builds brand-specific (customer_message, brand_reply) pairs from the raw
Customer Support on Twitter dataset.

This project is intentionally real-data-only. It requires the Kaggle TWCS
file to exist. There is no synthetic fallback in the production pipeline.
"""

import re
import random
import pandas as pd
from pathlib import Path

from . import config


def _clean(text: str) -> str:
    text = str(text or "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("\x00", " ")
    text = re.sub(r"@\w+", "", text)          # strip @mentions
    text = re.sub(r"http\S+", "", text)        # strip urls
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_real_pairs(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load twcs.csv, filter to BRAND_HANDLE, reconstruct first customer
    message -> first brand reply pairs, return a DataFrame with columns
    [thread_id, customer_text, brand_text]."""
    csv_path = Path(csv_path) if csv_path is not None else config.RAW_CSV_PATH
    df = pd.read_csv(csv_path, dtype=str)
    df["inbound"] = df["inbound"].astype(str).str.lower() == "true"

    by_id = df.set_index("tweet_id")

    brand_replies = df[(df["author_id"] == config.BRAND_HANDLE) & (~df["inbound"])]
    brand_replies = brand_replies.dropna(subset=["in_response_to_tweet_id"])

    rows = []
    for _, reply in brand_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        if parent_id not in by_id.index:
            continue
        parent = by_id.loc[parent_id]
        if isinstance(parent, pd.DataFrame):  # duplicate ids, skip ambiguous
            continue
        if not parent.get("inbound", False):
            continue  # only want genuine customer-initiated messages
        cust_text = _clean(str(parent["text"]))
        brand_text = _clean(str(reply["text"]))
        if len(cust_text) < 5 or len(brand_text) < 5:
            continue
        rows.append({
            "thread_id": reply["tweet_id"],
            "customer_text": cust_text,
            "brand_text": brand_text,
        })

    out = pd.DataFrame(rows).drop_duplicates(subset=["customer_text"])
    if config.SUBSAMPLE_N_THREADS and len(out) > config.SUBSAMPLE_N_THREADS:
        out = out.sample(config.SUBSAMPLE_N_THREADS, random_state=config.RANDOM_SEED)
    return out.reset_index(drop=True)


def get_pairs() -> pd.DataFrame:
    """Entry point used by the rest of the pipeline."""
    real_csv = config.resolve_raw_csv_path()
    if not real_csv.exists():
        raise FileNotFoundError(
            "No real TWCS dataset found. Place the Kaggle file at data/twcs.csv or "
            "data/raw/twcs.csv before running the pipeline."
        )
    print(f"[data_prep] Loading REAL data from {real_csv}")
    return load_real_pairs(real_csv)


if __name__ == "__main__":
    pairs = get_pairs()
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_PROCESSED / "brand_pairs.csv"
    pairs.to_csv(out_path, index=False)
    print(f"[data_prep] Wrote {len(pairs)} pairs -> {out_path}")
