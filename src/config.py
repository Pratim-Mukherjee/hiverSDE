"""
Central config for the support-agent pipeline.

Brand chosen: AmazonHelp
Why: one of the highest-volume, most consistently "brand-owned" accounts in the
Customer Support on Twitter dataset (author_id == "AmazonHelp" for outbound
messages), broad enough to have several recurring intent clusters (delivery,
refunds, account access, billing), and its replies follow a fairly
consistent internal style, which makes "grounded in how the brand has
historically resolved similar issues" a meaningful, checkable claim.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_RAW = DATA_DIR / "raw"
DATA_PROCESSED = DATA_DIR / "processed"
EVAL_DIR = ROOT / "eval"
REPORT_DIR = ROOT / "report"

BRAND_HANDLE = "AmazonHelp"          # author_id of the brand's outbound tweets
RAW_CSV_NAME = "twcs.csv"            # official Kaggle filename
RAW_CSV_CANDIDATES = [
    DATA_RAW / RAW_CSV_NAME,
    DATA_DIR / RAW_CSV_NAME,
    DATA_PROCESSED / RAW_CSV_NAME,
]


def resolve_raw_csv_path() -> Path:
    for candidate in RAW_CSV_CANDIDATES:
        if candidate.exists():
            return candidate
    return DATA_RAW / RAW_CSV_NAME


RAW_CSV_PATH = resolve_raw_csv_path()

# How many brand-customer conversation threads we actually work with.
# The assignment explicitly expects/encourages a subsample, not the full 3M rows.
SUBSAMPLE_N_THREADS = 4000
RANDOM_SEED = 13

# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------
# Chosen by reading ~150 raw AmazonHelp customer-first tweets by hand and
# clustering by surface complaint type. Kept deliberately small (7) rather
# than Banking77-style granular (77) because at 7 the classifier is legible,
# each intent maps to a distinct resolution/escalation policy, and reviewers
# can sanity check the whole taxonomy in one sitting. See DECISION_LOG.md.
INTENTS = [
    "delivery_delay_or_missing",     # order not arrived / stuck in transit
    "order_wrong_or_damaged",        # wrong item, broken, defective on arrival
    "refund_or_return_request",      # wants money back / return label
    "account_or_access_issue",       # login, password, locked account, app bug
    "billing_or_charge_dispute",     # unexpected/duplicate charge, price issue
    "cancellation_request",          # wants to cancel order/subscription
    "general_inquiry_or_other",      # praise, vague question, small talk, unclassifiable
]

# Keyword lexicon used for (a) weak-supervision pretraining labels and
# (b) an explainable fallback when the ML classifier is not confident.
# NOTE: weak labels are noisy by design -- they exist to bootstrap a model,
# not to serve as ground truth. Ground truth is the hand-labelled golden set.
INTENT_KEYWORDS = {
    "delivery_delay_or_missing": [
        "hasn't arrived", "has not arrived", "still hasn't", "never arrived",
        "where is my", "wheres my", "tracking", "delayed", "late delivery",
        "still waiting", "hasn't shipped", "not delivered", "missing package",
        "lost package", "stuck in", "no update on my order",
    ],
    "order_wrong_or_damaged": [
        "wrong item", "wrong size", "wrong color", "damaged", "broken",
        "defective", "not what i ordered", "missing parts", "arrived broken",
        "faulty", "doesn't work", "stopped working",
    ],
    "refund_or_return_request": [
        "refund", "return label", "money back", "send it back", "want a refund",
        "reimburse", "return this", "how do i return",
    ],
    "account_or_access_issue": [
        "can't log in", "cant log in", "login", "locked out", "password",
        "account is locked", "app keeps crashing", "won't let me sign in",
        "2fa", "verification code",
    ],
    "billing_or_charge_dispute": [
        "charged twice", "double charged", "overcharged", "wrong amount",
        "unexpected charge", "billed for", "charge on my card", "invoice",
        "price is wrong", "didn't authorize",
    ],
    "cancellation_request": [
        "cancel my order", "cancel subscription", "want to cancel", "cancel it",
        "stop my order", "cancel prime",
    ],
    "general_inquiry_or_other": [],  # catch-all, no keywords -> default bucket
}

# ---------------------------------------------------------------------------
# Escalation policy
# ---------------------------------------------------------------------------
# Intents/conditions that should NEVER be auto-closed by the agent, regardless
# of classifier confidence. Kept explicit and separate from the ML model so
# the safety boundary is auditable and doesn't drift with retraining.
ALWAYS_ESCALATE_INTENTS = {"billing_or_charge_dispute", "account_or_access_issue"}

# If the customer message trips any of these signals, escalate regardless of intent.
ESCALATION_TRIGGER_PHRASES = [
    "lawyer", "legal action", "sue", "fraud", "unauthorized charge",
    "hacked", "stole", "scam", "class action", "reporting you", "ftc",
    "kill myself", "suicide", "self harm",  # safety net; routes to human immediately
]

MIN_CLASSIFIER_CONFIDENCE_FOR_AUTOHANDLE = 0.55
# Tighten grounded-reply requirements to reduce ungrounded generation on noisy
# public-support text. Money/account intents are riskier and deserve stricter
# thresholds than delivery-status queries.
MIN_RETRIEVAL_SIMILARITY_FOR_GROUNDED_REPLY = 0.12
INTENT_SPECIFIC_GROUNDED_THRESHOLDS = {
    "billing_or_charge_dispute": 0.18,
    "account_or_access_issue": 0.18,
    "refund_or_return_request": 0.16,
    "order_wrong_or_damaged": 0.15,
    "cancellation_request": 0.14,
    "delivery_delay_or_missing": 0.12,
    "general_inquiry_or_other": 0.12,
}
