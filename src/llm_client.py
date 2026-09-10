"""
Thin optional wrapper around the Anthropic API.

The core pipeline (classification, retrieval, escalation) works with ZERO
API cost. This module is only used for:
  (a) an optional "polish" pass that rewrites a retrieved template more
      naturally while preserving its facts (off by default), and
  (b) the LLM-as-judge scorer in judge.py.

If ANTHROPIC_API_KEY is not set, both features degrade to deterministic
fallbacks (see judge.py's heuristic judge) rather than crashing, so the
whole repo still runs end-to-end for free.
"""

import os

_MODEL = "claude-sonnet-4-6"


def is_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def complete(prompt: str, max_tokens: int = 400) -> str:
    if not is_available():
        raise RuntimeError("ANTHROPIC_API_KEY not set; LLM features are disabled.")
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")
