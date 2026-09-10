"""
Reply-quality judge.

Two modes:
  - LLM judge (used if ANTHROPIC_API_KEY is set): scores each reply 1-5 on
    four axes using the rubric below and returns structured JSON.
  - Heuristic judge (always available, zero cost): a deterministic proxy
    scorer used (a) as the default so the whole eval harness runs for free,
    and (b) as one of the "judges" whose agreement with human labels we
    report, exactly like you'd report agreement for an LLM judge.

Whichever judge is used, judge_agreement.py compares its scores against a
human-labelled subset (see eval/golden_set_labeling_notes.md) and reports
agreement -- this is the "evidence the judge agrees with a human" the
assignment asks for.
"""

import json
import re
from dataclasses import dataclass, asdict
from . import llm_client

RUBRIC_PROMPT = """You are grading a customer-support agent's draft reply.

Customer message: {customer_text}
Agent's intent classification: {intent}
Agent's draft reply: {draft_reply}
Historical brand reply this was grounded on (evidence): {evidence_text}

Score the draft reply from 1 (very poor) to 5 (excellent) on each axis:
- relevance: does it address what the customer actually said?
- groundedness: is it consistent with the historical evidence (no invented facts/promises)?
- tone: is it polite, on-brand, appropriately empathetic?
- actionability: does the customer know what to do next?

Respond with ONLY compact JSON, no prose, in this exact shape:
{{"relevance": <1-5>, "groundedness": <1-5>, "tone": <1-5>, "actionability": <1-5>, "notes": "<one short sentence>"}}
"""


@dataclass
class JudgeScore:
    relevance: int
    groundedness: int
    tone: int
    actionability: int
    notes: str
    source: str  # "llm" or "heuristic"

    def overall(self) -> float:
        return (self.relevance + self.groundedness + self.tone + self.actionability) / 4.0

    def to_dict(self):
        return asdict(self)


def llm_judge(customer_text, intent, draft_reply, evidence_text) -> JudgeScore:
    prompt = RUBRIC_PROMPT.format(
        customer_text=customer_text, intent=intent,
        draft_reply=draft_reply, evidence_text=evidence_text or "(none)",
    )
    raw = llm_client.complete(prompt, max_tokens=200)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    data = json.loads(match.group(0)) if match else {}
    return JudgeScore(
        relevance=int(data.get("relevance", 3)),
        groundedness=int(data.get("groundedness", 3)),
        tone=int(data.get("tone", 3)),
        actionability=int(data.get("actionability", 3)),
        notes=str(data.get("notes", "")),
        source="llm",
    )


_ACTION_PHRASES = ["dm", "please send", "reach out", "let us know", "contact us", "reply with"]
_APOLOGY_PHRASES = ["sorry", "apolog", "understand", "appreciate"]


def heuristic_judge(customer_text, intent, draft_reply, evidence_text, grounded: bool) -> JudgeScore:
    """Deterministic proxy: cheap word-overlap + keyword checks. Not a substitute
    for real quality judgment, but a free, reproducible, always-available signal,
    and itself a data point we validate against human labels (see report)."""
    dl = draft_reply.lower()

    cust_words = set(re.findall(r"[a-z]+", customer_text.lower()))
    reply_words = set(re.findall(r"[a-z]+", dl))
    overlap = len(cust_words & reply_words) / max(1, len(cust_words))
    relevance = 5 if overlap > 0.25 else 4 if overlap > 0.12 else 2

    groundedness = 5 if grounded else 2

    tone = 4 + (1 if any(p in dl for p in _APOLOGY_PHRASES) else 0)
    tone = min(tone, 5)

    actionability = 5 if any(p in dl for p in _ACTION_PHRASES) else 3

    return JudgeScore(
        relevance=relevance, groundedness=groundedness, tone=tone, actionability=actionability,
        notes="heuristic: word-overlap + keyword proxy, no semantic understanding",
        source="heuristic",
    )


def judge_reply(customer_text, intent, draft_reply, evidence_text, grounded: bool) -> JudgeScore:
    if llm_client.is_available():
        try:
            return llm_judge(customer_text, intent, draft_reply, evidence_text)
        except Exception as e:
            print(f"[judge] LLM judge failed ({e}), falling back to heuristic.")
    return heuristic_judge(customer_text, intent, draft_reply, evidence_text, grounded)
