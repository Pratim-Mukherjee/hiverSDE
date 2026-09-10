# Decision Log

Non-obvious decisions made while building this, and why.

1. **Picked AmazonHelp over other brands.** High-volume, brand-owned handle
   (not a random reseller), consistent internal reply style — makes
   "grounded in historical resolution" a meaningful, checkable claim rather
   than noise.

2. **7 intents, not Banking77's 77.** At 7, each intent maps to a distinct
   escalation policy and a human reviewer can audit the whole taxonomy in
   minutes. Banking77-style granularity would need per-class escalation
   rules I can't actually justify for a support-Twitter dataset (it's built
   for banking chatbots, not general e-commerce support).

3. **Weak-supervision (keyword) labels to bootstrap the classifier, not
   hand-labels.** Hand-labeling is reserved entirely for the golden
   *evaluation* set, so training and evaluation never share a labeling
   process — otherwise the classifier is trivially "aligned" with its own
   grader.

4. **Hybrid classifier: ML primary, keyword fallback on low confidence, not
   the reverse.** ML generalizes past exact keyword matches; the keyword
   rule is a conservative safety net, not the main system, because keyword
   rules are brittle to paraphrase (see Failure Mode #1 in the report).

5. **Retrieval-templated replies instead of LLM-generated-from-scratch
   replies as the default.** A retrieved-and-adapted historical reply is a
   falsifiable claim ("this is literally what the brand said last time");
   an LLM asked to "write something similar" is not grounded in anything
   checkable. LLM polish is optional, off by default, and explicitly
   forbidden from adding new facts in its own prompt.

6. **Placeholder-scrubbing retrieved replies (`_templatize`)** so another
   customer's order number/email doesn't leak into a new customer's draft.
   This is a data-hygiene/privacy decision as much as a quality one.

7. **Escalation is rule-based, not learned.** It's the one safety-critical
   decision in the system; I wanted it legible and independent of model
   retraining drift, with an explicit, editable allow-list
   (`ALWAYS_ESCALATE_INTENTS`, `ESCALATION_TRIGGER_PHRASES`) rather than a
   black-box threshold on a classifier's own softmax.

8. **Two separate confidence gates: intent-confidence threshold AND
   retrieval-similarity threshold**, either of which alone forces
   escalation. A confident-but-ungrounded classification (weird phrasing,
   no similar precedent) is exactly as unsafe to auto-send as an
   unconfident one.

9. **Never auto-handle with an empty draft**, enforced as a hard safety net
   in `agent.py` even though the escalation logic should already prevent
   this — defense in depth on the one thing that must never happen (an
   auto-send with nothing to send).

10. **Trivial and simple baselines both reuse the same retriever/data code
    as the real agent** (not separate mocked-up baselines) so the
    comparison isolates the *policy/confidence logic*, not incidental
    differences in data handling.

11. **Heuristic judge as a first-class fallback, not just an error path.**
    Since the assignment must be reproducible without a paid API, the
    heuristic judge is explicitly built, documented as a proxy (word
    overlap + keyword checks), and its limitations are stated in the report
    rather than hidden — including that it partly rewards template
    boilerplate.

12. **Golden-set stratified-then-topped-up sampling**, not pure random or
    pure per-class-balanced. Pure random under-samples rare intents; pure
    per-class balance would misrepresent real traffic mix. The hybrid gives
    enough coverage of rare intents to evaluate them at all, while keeping
    some rows that reflect actual traffic shape.

13. **Subsampling to ~4000 threads, not the full ~3M-row dataset.** The
    assignment explicitly expects/encourages this; full-scale processing
    buys nothing for a triage/classification system whose signal saturates
    long before 3M rows, and it would blow past the 15-minute reproduction
    budget.

14. **Escalation trigger-phrase list intentionally includes self-harm
    phrases**, routing straight to a human with no auto-reply — a support
    bot must never attempt to auto-resolve a message containing that
    language, independent of anything else in the pipeline.

15. **Real-data-only execution is the default and expected mode**. The
    project refuses to silently pretend synthetic rows are real support
    traffic, and it fails fast if the Kaggle file is absent — this is a
    better engineering decision for auditability than masking a missing
    dataset with demo data.
