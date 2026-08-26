# AI Conversation Quality — Golden Dataset & Judge Evaluation Framework

A real, tested Python project pairing a hand-checkable "golden dataset"
with a judge-evaluation framework — built to close a specific gap for
Zendesk's "Machine Learning Engineering Intern" posting, whose core ask
(curate golden datasets, build LLM-as-Judge evaluation logic, compare
automated scores against human feedback and iterate) wasn't covered by
anything in my existing project portfolio. My prior projects evaluate
*models I built* honestly; none of them evaluate *another AI's output*
the way this posting asks for.

## What this is (read before citing anywhere)

**There is no real Zendesk (or any company's) conversation data here.**
`src/golden_dataset.py` generates synthetic customer-support
conversations across three quality tiers (good / borderline / bad), each
with a human-style annotation: a label plus a specific, tier-appropriate
reason — 60 conversations total, deliberately small, because a golden
set's value is in every label being individually defensible, not in raw
volume.

**There is no live LLM-as-Judge call.** I checked rather than assumed:
this build environment has no LLM API key and no Anthropic/OpenAI SDK
installed. `src/judge.py` defines a `Judge` interface shaped exactly like
a real LLM-as-Judge call (`transcript -> (label, reasoning)`), with two
implementations:

- **`HeuristicJudge`** — a real, deterministic, keyword/pattern-based
  judge, actually implemented and actually evaluated, not a placeholder.
- **`PromptedLLMJudge`** — honestly left unimplemented. It raises
  `NotImplementedError` explaining exactly why, and its docstring
  contains the actual evaluation prompt a live LLM-as-Judge call would
  use (the prompt design itself is real evaluation-logic work,
  independent of whether the call can run here). Swapping in a live
  Braintrust/LLM call means implementing this one method — nothing else
  in the framework (the golden dataset, the agreement scoring, the
  disagreement review) needs to change.

If asked in an interview: I haven't used Braintrust and this isn't a
live LLM-as-Judge system. This project demonstrates the underlying
skills the posting actually asks for — curating a golden set with
defensible labels, building a judge with a clean interface for
comparing automated scores against ground truth, and — most
importantly — actually iterating on the judge instead of reporting the
first number I got, on a system I could build and verify myself.

## The honest finding — this is the part that matters most

The first version of `HeuristicJudge` scored **100% agreement** against
the golden dataset. Following the same rule I've applied to every model
in this portfolio — a suspiciously perfect result gets investigated, not
reported — I checked why, by testing the judge against a second, hand-
written set of paraphrases (`src/paraphrase_holdout.py`) that a human
would label identically but that don't reuse the golden set's exact
template wording (e.g. "power cycle the machine" instead of "restart
your computer").

Agreement on that held-out paraphrase set: **25% (2 of 8)** — down from
100%. The judge's keyword markers had been implicitly tuned to the exact
phrasing the golden-dataset templates happen to use, not to the
underlying concept of "does this response resolve, deflect, or generi-
cally respond." One specific bug was real and fixed along the way
(`"restart your computer"` didn't match the template's actual
`"restarting"` wording — a literal string-match miss, not a conceptual
one); after fixing it, golden-set agreement legitimately reached 100%.
But that alone didn't mean the judge generalized — the paraphrase test is
what actually revealed that.

**This is disclosed, not hidden, and it's locked in as a permanent
regression test** (`test_generalization_to_paraphrases_is_the_real_test`
and `test_golden_set_score_is_not_evidence_of_generalization` in
`tests/test_judge.py`) — specifically to prevent the mistake of reporting
only the flattering number from happening again, in this project or the
next one.

This is exactly the "iterate to perfection" loop the posting describes —
compare the automated score against something closer to real human
judgment, find that it doesn't hold up, and treat that as the actual
result rather than a failure to hide. A real LLM-as-Judge would face the
same risk in a different form (overfitting to the exact phrasing of a
prompt-engineering iteration loop rather than the underlying quality
signal), which is exactly why held-out, differently-phrased evaluation
data matters for any judge, heuristic or LLM-based.

## What this models

- **`src/golden_dataset.py`** — synthetic support-conversation generator,
  three quality tiers, each conversation carrying a human-style
  `gold_label` and `gold_reason`.
- **`src/paraphrase_holdout.py`** — a second, independently-written set of
  8 labeled examples using different phrasing, used specifically to test
  whether the judge generalizes rather than memorizes.
- **`src/judge.py`** — the `Judge` interface, `HeuristicJudge` (real,
  implemented, evaluated), and `PromptedLLMJudge` (honestly stubbed, with
  the real evaluation prompt documented).
- **`src/evaluate.py`** — the evaluation harness: agreement rate,
  confusion matrix, and every individual disagreement case with both the
  judge's and the gold reasoning shown side by side.

## Verification

19 tests (`pytest tests/ -v`), including:

- Golden-dataset structural checks (tier balance, unique IDs, no
  overlapping response templates across tiers, distinct reasoning text
  per tier).
- `HeuristicJudge` scores ≥95% on the golden set it was built against.
- The generalization test: agreement on the paraphrase holdout is
  asserted to stay in a realistic 15–45% band — not just a floor, so
  that fixing this properly (e.g. real NLP features instead of exact
  keyword matches) would need a deliberate change to this test, not an
  accidental one.
- A dedicated test asserting the gap between golden-set and holdout
  agreement stays large, as an explicit guard against reporting only the
  flattering number again.
- `PromptedLLMJudge` raises a clear, specific `NotImplementedError` and
  its docstring contains the real evaluation prompt.

## Running it

```bash
pip install -r requirements.txt   # no dependencies beyond pytest — stdlib only
python3 -m src.evaluate            # runs HeuristicJudge against the golden set, prints agreement + disagreements
pytest tests/ -v                   # 19 tests
```

## What this doesn't demonstrate

This project doesn't use real conversation data, doesn't call a live LLM
as a judge (no API access in this environment — verified, not assumed),
and doesn't integrate with Braintrust or any real eval platform. The
`HeuristicJudge`'s 25% holdout-generalization score is a genuine,
disclosed limitation of a keyword-based approach, not a demonstration
that this method is production-ready — it demonstrates the pattern the
posting actually asks for (build a golden set, build a judge, measure it
against something closer to ground truth, and treat a bad result as
signal to iterate on rather than something to hide) on a system I could
construct and verify myself.
