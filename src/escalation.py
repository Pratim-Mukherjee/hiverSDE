"""
Decides AUTO_HANDLE vs ESCALATE_TO_HUMAN, always with a stated reason.

Deliberately rule-based (not learned) -- this is a safety-relevant decision
and needs to be auditable/adjustable without retraining a model. See
DECISION_LOG.md for why we didn't make this an ML classifier.
"""

from dataclasses import dataclass
from . import config


@dataclass
class EscalationDecision:
    action: str          # "AUTO_HANDLE" or "ESCALATE_TO_HUMAN"
    reason: str


def decide(customer_text: str, intent: str, intent_confidence: float,
           reply_is_grounded: bool) -> EscalationDecision:
    text_lower = customer_text.lower()

    for phrase in config.ESCALATION_TRIGGER_PHRASES:
        if phrase in text_lower:
            return EscalationDecision(
                "ESCALATE_TO_HUMAN",
                f"Trigger phrase detected ('{phrase}') -- routed to human regardless of intent/confidence.",
            )

    if intent in config.ALWAYS_ESCALATE_INTENTS:
        return EscalationDecision(
            "ESCALATE_TO_HUMAN",
            f"Intent '{intent}' is policy-flagged as always-escalate "
            "(touches account security or money movement -- auto-reply risk too high).",
        )

    if intent_confidence < config.MIN_CLASSIFIER_CONFIDENCE_FOR_AUTOHANDLE:
        return EscalationDecision(
            "ESCALATE_TO_HUMAN",
            f"Intent classifier confidence {intent_confidence:.2f} is below the "
            f"{config.MIN_CLASSIFIER_CONFIDENCE_FOR_AUTOHANDLE} auto-handle threshold.",
        )

    if not reply_is_grounded:
        return EscalationDecision(
            "ESCALATE_TO_HUMAN",
            "No sufficiently similar historical resolution was retrieved, so no grounded "
            "draft could be produced -- unsafe to auto-send an ungrounded reply.",
        )

    return EscalationDecision(
        "AUTO_HANDLE",
        f"Intent '{intent}' recognized with confidence {intent_confidence:.2f}, "
        "a grounded historical reply template was found, and no escalation triggers fired.",
    )
