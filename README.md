# AI Conversation Quality — Golden Dataset & Judge Evaluation Framework

A tested Python project pairing a hand-checkable "golden dataset" with a
judge-evaluation framework. It covers three parts of evaluating *another
AI's* output: curating a golden dataset with defensible labels, building
LLM-as-Judge-style evaluation logic, and comparing automated scores
against human-style labels and iterating on the result. My other
projects evaluate models I built; this one evaluates conversation
quality.

## Scope

**There is no real company conversation data here.**
`src/golden_dataset.py` generates synthetic customer-support
conversations across three quality tiers (good / borderline / bad), each
with a human-style annotation: a label plus a specific, tier-appropriate
reason — 60 conversations total, deliberately small, because a golden
set's value is in every label being individually defensible, not in raw
volume.

**There is no live LLM-as-Judge call.** I checked: my build environment
has no LLM API key and no Anthropic/OpenAI SDK installed.
`src/judge.py` defines a `Judge` interface shaped like a real
LLM-as-Judge call (`transcript -> (label, reasoning)`), with two
implementations:

- **`HeuristicJudge`** — a deterministic, keyword/pattern-based judge,
  implemented and evaluated, not a placeholder.
- **`PromptedLLMJudge`** — left unimplemented. It raises
  `NotImplementedError` explaining why, and its docstring contains the
  evaluation prompt a live LLM-as-Judge call would use (the prompt
  design is evaluation-logic work, independent of whether the call can
  run here). Swapping in a live LLM call means implementing this one
  method; nothing else in the framework (the golden dataset, the
  agreement scoring, the disagreement review) needs to change.

## Key finding: the first judge didn't generalize

The first version of `HeuristicJudge` scored **100% agreement** against
the golden dataset. A suspiciously perfect result gets investigated, so
I tested the judge against a second, hand-written set of paraphrases
(`src/paraphrase_holdout.py`) that a human would label identically but
that don't reuse the golden set's exact template wording (e.g. "power
cycle the machine" instead of "restart your computer").

Agreement on that held-out paraphrase set: **25% (2 of 8)** — down from
100%. The judge's keyword markers had been implicitly tuned to the exact
phrasing the golden-dataset templates happen to use, not to the
underlying concept of "does this response resolve, deflect, or
generically respond." One specific bug was found and fixed along the way
(`"restart your computer"` didn't match the template's actual
`"restarting"` wording — a literal string-match miss, not a conceptual
one); after fixing it, golden-set agreement legitimately reached 100%.
But that alone didn't mean the judge generalized; the paraphrase test is
what revealed that.

The result is locked in as a permanent regression test
(`test_generalization_to_paraphrases_is_the_real_test` and
`test_golden_set_score_is_not_evidence_of_generalization` in
`tests/test_judge.py`), so the flattering golden-set number can't be
reported on its own in this project or the next one.

This is the "iterate to perfection" loop in practice: compare the
automated score against something closer to human judgment, find that it
doesn't hold up, and treat that as the actual result. A real
LLM-as-Judge would face the same risk in a different form (overfitting
to the exact phrasing of a prompt-engineering iteration loop rather than
the underlying quality signal), which is why held-out, differently-
phrased evaluation data matters for any judge, heuristic or LLM-based.

## What this models

- **`src/golden_dataset.py`** — synthetic support-conversation generator,
  three quality tiers, each conversation carrying a human-style
  `gold_label` and `gold_reason`.
- **`src/paraphrase_holdout.py`** — a second, independently-written set of
  8 labeled examples using different phrasing, used specifically to test
  whether the judge generalizes rather than memorizes.
- **`src/judge.py`** — the `Judge` interface, `HeuristicJudge` (real,
  implemented, evaluated), and `PromptedLLMJudge` (stubbed, with
  the evaluation prompt documented).
- **`src/evaluate.py`** — the evaluation harness: agreement rate,
  confusion matrix, and every individual disagreement case with both the
  judge's and the gold reasoning shown side by side.

## Testing

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

## Notes

The project uses no real conversation data, calls no live LLM as a
judge (no API access in my environment), and doesn't integrate with an
eval platform. The `HeuristicJudge`'s 25% holdout-generalization score
is a measured limitation of a keyword-based approach, not a claim that
the method is production-ready; the project's value is the pattern:
build a golden set, build a judge, measure it against something closer
to ground truth, and treat a bad result as a signal to iterate.
