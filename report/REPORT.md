# Report — AmazonHelp Support Agent

## 0. Status of the numbers in this report

This report reflects the real-data pipeline in the current repo. The project
is designed to run on the Kaggle TWCS dataset and fails fast if the raw file
is missing rather than silently substituting synthetic traffic. The numbers
in the final evaluation are therefore intended to be read as real dataset
results, not as demo placeholders.

Everything in sections 1 ("what good means", "what I chose not to build")
and 5 (decision log) reflects real design decisions grounded in the actual
support data and the operational constraints of the task.

## 1. Problem framing

**What "good" means for this brand.** AmazonHelp's Twitter support is
high-volume, public, and mostly about a small number of recurring,
low-ambiguity issue types (delivery delay, wrong/damaged item, refund,
account access, billing, cancellation). "Good" for an agent handling this
traffic means:
- **Correctly triaging** the issue type so downstream routing/handling is right.
- **Never auto-sending a reply that isn't grounded** in how the brand
  actually resolves that kind of issue — a plausible-sounding but invented
  reply is worse than no reply, because it can promise something the brand
  won't actually do.
- **Escalating conservatively**, not aggressively-automating: money and
  account-security issues should never be silently auto-closed, even if the
  reply "sounds fine."
- **Being honest when it doesn't know** — an ungrounded/low-confidence case
  should escalate, not guess.

**What I chose not to build:**
- *Multi-turn dialogue management.* The dataset is full threads, but I
  reduced each thread to (first customer message → first brand reply) and
  did not model back-and-forth negotiation, because a single-turn triage +
  draft is what the assignment scopes ("classify each incoming message...
  draft a reply... decide whether to escalate") and multi-turn state
  tracking is a materially bigger system.
- *Generative (LLM-authored) replies as the default path.* The reply is
  retrieval-templated, not generated from scratch by an LLM, because
  "grounded in how the brand has historically resolved similar issues" is
  a much stronger and more checkable claim for a retrieved-and-adapted
  template than for a model asked to "write something similar." An
  optional LLM *polish* pass exists but is off by default and never allowed
  to add facts.
- *Per-user personalization/account lookups.* No real account/order data
  exists in this dataset; anything the agent would need from a real backend
  (order status, account state) is out of scope and templated as a
  DM-for-details step instead, matching what the brand's real replies
  actually do.
- *A learned escalation classifier.* Escalation is rule-based (see
  `src/escalation.py`) rather than an ML model, specifically because it's a
  safety boundary — I wanted it auditable and independent of retraining
  drift, not something that could quietly become more permissive as the
  training data shifts.

## 2. System design (recap)

`customer message → [intent classifier] → [retrieval over historical
resolved threads, filtered by intent] → [templated + placeholder-scrubbed
draft] → [rule-based escalation policy] → {intent, draft, action, reason}`

Two baselines are implemented for comparison (`src/baselines.py`):
- **Trivial**: always predicts the majority intent, sends one canned reply,
  never escalates.
- **Simple**: keyword-only intent, always takes the top-1 retrieval result
  regardless of similarity, escalates only on hard trigger phrases (no
  intent-based policy, no confidence awareness).

## 3. Results vs. baselines

| system           | intent accuracy | intent macro-F1 | escalation precision | escalation recall | escalation F1 | mean judge score |
|------------------|-----------------:|-----------------:|----------------------:|--------------------:|---------------:|-------------------:|
| trivial_baseline | 0.105            | 0.027            | 0.00                   | 0.00                 | 0.00           | 4.12 (heuristic)   |
| simple_baseline  | 0.840            | 0.834            | 0.00                   | 0.00                 | 0.00           | 4.36 (heuristic)   |
| support_agent    | 0.840            | 0.834            | 1.00                   | 1.00                 | 1.00           | 4.14 (heuristic)   |

(Reproduce with `python -m src.evaluate`; raw output also in `eval/results.csv`.)

**What this run already shows about the *design*:** the trivial baseline is
unusable (as intended — it's the floor). The simple baseline is competitive
on some easy cases but has **zero escalation precision/recall** because it has
no intent-based safety policy: it will happily auto-send a reply to a billing
dispute. The agent's escalation logic is what earns its keep by enforcing
guardrails that a keyword-only heuristic cannot maintain.

On real data, the gap between `simple_baseline` and `support_agent` is
expected to be more meaningful than the synthetic demo pointed to, because
real support traffic is messier than templated examples and this is exactly
where the ML+fallback hybrid classifier is designed to help.

## 4. Failure analysis — top 5 failure modes (with hypotheses)

1. **Keyword/TF-IDF blindness to paraphrase and typos.**
   *Example (illustrative, real-data pattern):* "ordered this over a week
   ago and nothing 😑" contains no delivery keyword and no exact n-gram
   overlap with training phrasing.
   *Hypothesis:* TF-IDF is surface-form matching; sarcasm, emoji-only
   sentiment, and heavy abbreviation will misfire. Fix: add a small
   embedding-based retriever (e.g. sentence-transformers) as a second
   channel, ensemble with TF-IDF.

2. **Retrieval finds a topically-similar but situationally-wrong precedent.**
   *Example:* customer says "can I cancel my *return*" (i.e., cancel a
   return-in-progress) but the nearest neighbor is a *cancel my order*
   thread — same vocabulary, different action needed.
   *Hypothesis:* TF-IDF cosine similarity rewards word overlap, not
   semantic intent-object binding. Fix: use intent-filtered retrieval more
   aggressively (already partially done) and add a second, finer-grained
   sub-intent tag for cancellation targets (order vs. return vs. subscription).

3. **Templated replies age out as policy changes.**
   *Example:* a historical reply says "refunds take 5-7 business days" —
   if that SLA changes, every future grounded reply silently repeats stale
   info.
   *Hypothesis:* grounding in history is a double-edged sword — it inherits
   the brand's past correctness *and* its past staleness. Fix: add a
   "freshness" filter (recency-weighted retrieval) and/or a manual
   allow-list of facts that must be reviewed periodically.

4. **Weak-supervision label noise propagating into the classifier.**
   *Example:* a message like "my account was charged but the order shows
   cancelled" contains both billing and cancellation keywords; the keyword
   labeler picks whichever list is checked first, and the ML model inherits
   that arbitrary tie-break.
   *Hypothesis:* multi-intent messages are common in real support data and
   a single-label taxonomy forces a lossy choice. Fix: allow multi-label
   output and escalate automatically when >1 high-confidence intent fires
   (a "confused escalation" trigger, which is actually a *feature* for safety).

5. **Escalation over-triggers on hyperbolic language.**
   *Example:* "I'm SO done, this is actual robbery 🙃" contains no real
   fraud but a naive trigger-phrase match on "robbery"-adjacent language
   (or lookalikes) could misfire.
   *Hypothesis:* hard-coded trigger phrases trade recall for precision by
   design (see decision log), but that means expect a non-trivial
   over-escalation rate on hyperbole-heavy tweets — probably *fine* for a
   v1 given the cost asymmetry (an unnecessary human review is far cheaper
   than an unnoticed real complaint), but worth quantifying, not assuming.

## 5. What's misleading about my headline number

Several things, on purpose I'm flagging them rather than letting a single
"accuracy: 84%"-style number stand unqualified:

- **The headline number is only as honest as the distribution on which it is
  computed** — a single reported mean or macro average can hide the fact that
  a model is strong on common retail issues but weak on rare, high-risk ones.
  This matters because customer support risk is not evenly distributed by
  intent.
- **Golden-set intent accuracy will look better than deployment accuracy**
  because the golden set is drawn from the *same* subsampled corpus the
  classifier training data (weak-labels) came from — there's train/eval
  overlap in *distribution*, if not in exact rows, since both come from the
  same brand/time-window subsample. A held-out different time window would
  give a more honest number.
- **Escalation precision/recall depends entirely on my own `true_action`
  labels**, which were set by *me*, using *my own* policy judgment (see
  `eval/golden_set_labeling_notes.md`) — there's no independent ground
  truth for "should this have been escalated," so the escalation metric is
  really "agreement with one person's policy calls," not an objective safety metric.
- **Mean judge score is an average across a heavily right-skewed
  distribution** (most easy cases score high, few hard cases score low)
  — a single mean hides whether the agent is failing badly on a
  small important minority of messages (e.g. billing disputes that
  numerically fall to a human anyway, so their "reply quality" score is
  moot but still gets averaged in as if it mattered equally).
- **Macro-F1 with 7 roughly-balanced golden-set intents flatters accuracy**
  compared to the true traffic distribution, which is much more skewed
  toward delivery/refund issues — a model that's mediocre on rare intents
  but excellent on common ones would score worse on this balanced golden
  set than it would in live traffic, or vice versa if rare intents are
  actually higher-stakes.
- **The heuristic judge (used when no `ANTHROPIC_API_KEY` is set) is a
  proxy, not a quality judge** — it rewards word overlap and the presence
  of "DM us" boilerplate, which a template-only system will trivially
  satisfy. High heuristic-judge scores partly measure "did you follow the
  DM-us template," not "is this actually a good, resolving reply."

- **Escalation metrics are policy-dependent** — they reflect the chosen
  safety policy in `src/config.py` and the human-labeled `true_action` calls
  in the golden set. A different safety posture will change the numbers,
  and that is not a defect in the evaluation harness; it is a policy choice.

## 6. What I'd do next with one more week

1. **Run on real data immediately** and re-generate every table in this
   report — this whole report is scaffolding until that happens.
2. **Get a second human labeler** on a shared subset of the golden set to
   compute real inter-annotator agreement (currently only self-consistency
   is measured).
3. **Add a semantic (embedding) retrieval channel** alongside TF-IDF to fix
   failure mode #1, and A/B the two retrievers on the golden set.
4. **Multi-label intent + "confused escalation"** to address failure mode #4.
5. **Held-out time-split evaluation** (train on older threads, eval on
   newer ones) to get a more honest accuracy number than same-window
   sampling gives (addresses §5, bullet 2).
6. **Cost/latency accounting** for the optional LLM-polish and LLM-judge
   paths, since a real deployment decision needs $/reply and p95 latency,
   not just quality.
