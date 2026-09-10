"""
Measures how well the judge (LLM or heuristic) agrees with a human rater.

Process:
  1. Take a random subsample (default 30) of golden_eval.csv rows.
  2. Run the real agent on each to get a draft reply.
  3. A human rates each draft 1-5 on the same 4 axes as judge.RUBRIC_PROMPT
     and saves it to eval/human_judge_ratings.csv (columns:
     thread_id, human_relevance, human_groundedness, human_tone, human_actionability).
  4. This script loads both, computes per-axis correlation (Spearman) and
     overall-score mean absolute error, and reports them -- this is the
     "evidence the judge agrees with a human" required by the assignment.

Run:
    python -m eval.human_agreement            # step: generates the rating sheet
    ... human fills in eval/human_judge_ratings.csv ...
    python -m eval.human_agreement --score     # step: computes agreement
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import spearmanr

from src import config
from src.evaluate import build_agent_and_baselines, load_golden
from src.judge import judge_reply

N_SUBSAMPLE = 30
RATING_SHEET = config.EVAL_DIR / "human_judge_ratings_TEMPLATE.csv"
RATING_FILLED = config.EVAL_DIR / "human_judge_ratings.csv"


def generate_sheet():
    golden = load_golden()
    sample = golden.sample(min(N_SUBSAMPLE, len(golden)), random_state=config.RANDOM_SEED)
    agent, _, _ = build_agent_and_baselines()

    rows = []
    for _, row in sample.iterrows():
        out = agent.handle(row["customer_text"]).to_dict()
        evidence_text = out["evidence"][0]["historical_brand_text"] if out["evidence"] else ""
        js = judge_reply(row["customer_text"], out["intent"], out["draft_reply"],
                          evidence_text, grounded=out["reply_grounded"])
        rows.append({
            "thread_id": row["thread_id"],
            "customer_text": row["customer_text"],
            "draft_reply": out["draft_reply"],
            "judge_relevance": js.relevance, "judge_groundedness": js.groundedness,
            "judge_tone": js.tone, "judge_actionability": js.actionability,
            "judge_source": js.source,
            "human_relevance": "", "human_groundedness": "", "human_tone": "", "human_actionability": "",
        })
    pd.DataFrame(rows).to_csv(RATING_SHEET, index=False)
    print(f"Wrote {RATING_SHEET}. Fill in human_* columns, save as {RATING_FILLED}, then re-run with --score.")


def score():
    df = pd.read_csv(RATING_FILLED)
    axes = ["relevance", "groundedness", "tone", "actionability"]
    print(f"{'axis':<15}{'spearman_r':>12}{'mean_abs_diff':>16}")
    for axis in axes:
        j, h = df[f"judge_{axis}"], df[f"human_{axis}"]
        r, _ = spearmanr(j, h)
        mad = (j - h).abs().mean()
        print(f"{axis:<15}{r:>12.2f}{mad:>16.2f}")

    j_overall = df[[f"judge_{a}" for a in axes]].mean(axis=1)
    h_overall = df[[f"human_{a}" for a in axes]].mean(axis=1)
    r_overall, _ = spearmanr(j_overall, h_overall)
    print(f"\noverall spearman r = {r_overall:.2f}, mean abs diff = {(j_overall - h_overall).abs().mean():.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    score() if args.score else generate_sheet()
