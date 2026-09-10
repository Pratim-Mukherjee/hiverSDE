# Golden Evaluation Set — Sampling & Labeling Notes

**Size:** 200 examples (within the required 150-250).

**Source:** first customer message → first AmazonHelp reply pairs, extracted
from `data/raw/twcs.csv` by `src/data_prep.py`, subsampled to
`SUBSAMPLE_N_THREADS` threads before golden-set sampling (see `src/config.py`).

**Sampling method** (`scripts/make_golden_set.py`):
1. Every candidate thread is auto-tagged with the cheap keyword rule
   (`src/intents.keyword_label`).
2. We take up to ~28 examples per intent bucket (200 / 7) to get roughly even
   intent coverage — without this, the raw distribution is dominated by
   `delivery_delay_or_missing` and `general_inquiry_or_other`, and a
   random sample would barely exercise rarer intents like
   `cancellation_request`.
3. We top up to 200 with a uniform random draw from whatever wasn't already
   selected, specifically so the set isn't *entirely* shaped by what the
   keyword rule already agrees with — some rows are there because they're
   representative of overall traffic, not because they're "clean" examples.
4. Final order is shuffled so a human labeler isn't primed by intent-grouping.

**Labeling process:**
- One person (me) manually read each customer message + the brand's actual
  historical reply as context, and assigned:
  - `true_intent`: the correct label from the 7-intent taxonomy, overriding
    `suggested_intent` where the keyword rule was wrong.
  - `true_action`: `AUTO_HANDLE` or `ESCALATE_TO_HUMAN`, applying the same
    policy a support lead would: if the issue is unambiguous, low-risk,
    and has a standard resolution → AUTO_HANDLE; if it touches money,
    account security, contains distress/legal language, or is too vague
    to resolve without more info → ESCALATE_TO_HUMAN.
- ~15% of rows (30) were independently re-labeled a second time (blind to
  the first pass) to compute intra-labeler consistency, reported in the
  report's "what's misleading about my headline number" section — with a
  single labeler, this is a *consistency* check, not inter-annotator
  agreement, and is reported as such.

**Known limitations of this golden set (see report for full discussion):**
- Single labeler → no true inter-annotator agreement, only self-consistency.
- Twitter-support text is short and often under-specified; some
  `true_action` calls are genuinely borderline and were made using judgment,
  not a bright-line rule.
- The set reflects the same time period / product mix as the source dump,
  so it may not generalize to policy changes or seasonal issue spikes
  (e.g. holiday shipping delays).
