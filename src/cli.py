"""Explicit CLI for the support-agent pipeline.

Cross-platform usage from the repo root:
    python -m src.cli prep
    python -m src.cli golden
    python -m src.cli evaluate
    python -m src.cli agreement --score
    python -m src.cli all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_python(args: list[str], cwd: Path | None = None) -> None:
    subprocess.run([sys.executable, *args], cwd=str(cwd or ROOT), check=True)


def command_prep(args: argparse.Namespace) -> None:
    dataset = Path(args.dataset).resolve() if args.dataset else None
    if dataset is not None:
        print(f"[cli] Using dataset override: {dataset}")
        from src import config
        config.RAW_CSV_PATH = dataset
    _run_python(["-m", "src.data_prep"], cwd=ROOT)


def command_golden(args: argparse.Namespace) -> None:
    script = ROOT / "scripts" / "make_golden_set.py"
    _run_python([str(script)], cwd=ROOT)
    print("[cli] golden set regenerated from the real TWCS data; no synthetic rows are used.")


def command_review(args: argparse.Namespace) -> None:
    script = ROOT / "scripts" / "human_gold_review.py"
    _run_python([str(script)], cwd=ROOT)
    print("[cli] human-review template created; fill in true_intent / true_action in eval/golden_eval_reviewed.csv before final evaluation.")


def command_evaluate(args: argparse.Namespace) -> None:
    _run_python(["-m", "src.evaluate"], cwd=ROOT)


def command_agreement(args: argparse.Namespace) -> None:
    cmd = ["-m", "eval.human_agreement"]
    if args.score:
        cmd.append("--score")
    _run_python(cmd, cwd=ROOT)


def command_install(args: argparse.Namespace) -> None:
    print("[cli] Installing project dependencies into the active Python environment...")
    _run_python(["-m", "pip", "install", "-r", "requirements.txt"], cwd=ROOT)


def command_sample(args: argparse.Namespace) -> None:
    from src.data_prep import get_pairs
    from src.intents import IntentClassifier
    from src.retrieval import ReplyRetriever
    from src.agent import SupportAgent

    pairs = get_pairs()
    if args.n <= 0:
        raise ValueError("--n must be positive")
    sample_df = pairs.sample(min(args.n, len(pairs)), random_state=13).reset_index(drop=True)
    clf = IntentClassifier().fit(pairs["customer_text"].tolist())
    retr = ReplyRetriever(
        pairs["customer_text"].tolist(),
        pairs["brand_text"].tolist(),
        corpus_intents=[clf.predict_one(t).intent for t in pairs["customer_text"]],
    )
    agent = SupportAgent(clf, retr)

    print(f"\n[cli] Sample run on {len(sample_df)} rows from the real dataset\n")
    for idx, row in sample_df.iterrows():
        out = agent.handle(row["customer_text"]).to_dict()
        print(f"{idx + 1}. intent={out['intent']} | action={out['action']} | confidence={out['intent_confidence']:.3f}")
        print(f"   customer: {row['customer_text']}")
        print(f"   draft: {out['draft_reply']}")
        print()


def command_all(args: argparse.Namespace) -> None:
    if args.dataset:
        print(f"[cli] Dataset override: {args.dataset}")
    if not args.skip_prep:
        command_prep(args)
    if not args.skip_golden and not (ROOT / "eval" / "golden_eval.csv").exists():
        command_golden(args)
    command_evaluate(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AmazonHelp support-agent pipeline for the TWCS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prep", help="Build processed brand pairs from the raw dataset.")
    prep.add_argument("--dataset", type=str, help="Override dataset path (for example data/twcs.csv)")
    prep.set_defaults(func=command_prep)

    golden = subparsers.add_parser("golden", help="Generate a real-data golden-set candidate file.")
    golden.set_defaults(func=command_golden)

    review = subparsers.add_parser("review", help="Generate a human-reviewed golden-set template for final submission labeling.")
    review.set_defaults(func=command_review)

    evaluate = subparsers.add_parser("evaluate", help="Run metrics vs. the golden set.")
    evaluate.set_defaults(func=command_evaluate)

    agreement = subparsers.add_parser("agreement", help="Generate or score human judge-agreement ratings.")
    agreement.add_argument("--score", action="store_true", help="Score the completed human ratings file.")
    agreement.set_defaults(func=command_agreement)

    sample = subparsers.add_parser("sample", help="Run a small real-data sample through the agent.")
    sample.add_argument("--n", type=int, default=5, help="Number of sample rows to evaluate.")
    sample.set_defaults(func=command_sample)

    install = subparsers.add_parser("install", help="Install the Python dependencies into the active environment.")
    install.set_defaults(func=command_install)

    all_cmd = subparsers.add_parser("all", help="Run prep + evaluate in one command.")
    all_cmd.add_argument("--dataset", type=str, help="Override dataset path")
    all_cmd.add_argument("--skip-prep", action="store_true", help="Skip the data-prep step")
    all_cmd.add_argument("--skip-golden", action="store_true", help="Skip the golden-set generation check")
    all_cmd.set_defaults(func=command_all)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
