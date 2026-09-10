"""
End-to-end agent: one function in, one structured decision out.
"""

from dataclasses import dataclass, asdict
from . import config, llm_client
from .intents import IntentClassifier, IntentResult
from .retrieval import ReplyRetriever
from .escalation import decide, EscalationDecision


@dataclass
class AgentOutput:
    customer_text: str
    intent: str
    intent_confidence: float
    intent_used_fallback: bool
    draft_reply: str
    reply_grounded: bool
    grounding_reason: str
    evidence: list
    action: str
    escalation_reason: str

    def to_dict(self):
        return asdict(self)


class SupportAgent:
    def __init__(self, classifier: IntentClassifier, retriever: ReplyRetriever, use_llm_polish: bool = False):
        self.classifier = classifier
        self.retriever = retriever
        self.use_llm_polish = use_llm_polish and llm_client.is_available()

    def handle(self, customer_text: str) -> AgentOutput:
        intent_result: IntentResult = self.classifier.predict_one(customer_text)
        draft_info = self.retriever.draft_reply(customer_text, intent_filter=intent_result.intent)

        draft = draft_info["draft"]
        if draft and self.use_llm_polish:
            draft = self._polish(customer_text, draft)

        esc: EscalationDecision = decide(
            customer_text=customer_text,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            reply_is_grounded=draft_info["grounded"],
        )

        # Never auto-send with no draft at all.
        if esc.action == "AUTO_HANDLE" and not draft:
            esc = EscalationDecision("ESCALATE_TO_HUMAN", "Safety net: no draft text available despite grounded=True flag.")

        return AgentOutput(
            customer_text=customer_text,
            intent=intent_result.intent,
            intent_confidence=intent_result.confidence,
            intent_used_fallback=intent_result.used_fallback,
            draft_reply=draft or "[ESCALATED - no auto-draft produced]",
            reply_grounded=draft_info["grounded"],
            grounding_reason=draft_info["reason"],
            evidence=draft_info["evidence"],
            action=esc.action,
            escalation_reason=esc.reason,
        )

    def _polish(self, customer_text: str, templated_draft: str) -> str:
        prompt = (
            "You are lightly rewriting a customer-support reply template to sound natural. "
            "Do NOT add new facts, promises, or details beyond what's in the template. "
            "Keep placeholders like {ORDER_ID} exactly as-is.\n\n"
            f"Customer message: {customer_text}\n"
            f"Template reply: {templated_draft}\n\n"
            "Rewritten reply:"
        )
        try:
            return llm_client.complete(prompt).strip()
        except Exception:
            return templated_draft  # fail safe: keep the grounded template
