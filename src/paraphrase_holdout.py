"""
A held-out generalization set: hand-written paraphrases of the same three
quality tiers, using DIFFERENT wording than the templates in
golden_dataset.py.

Why this exists: the first version of `HeuristicJudge` scored 100%
agreement against the golden dataset, which — per the same
"suspicious perfection is a red flag" discipline used in every other
project in this portfolio — was investigated rather than reported as a
clean win. The judge's keyword markers were tuned against the exact
phrasing the golden-dataset templates happen to use (e.g. "restarting
your computer"), so a 100% score partly reflected memorizing that
phrasing, not genuinely recognizing deflection vs. resolution vs.
generic responses.

This module is the fix for the blind spot, not a hidden one: a second,
independently-written set of examples that a human annotator would label
the same way, but phrased differently from anything the judge's marker
list was built against. `evaluate_judge` is run against this set
separately in `tests/test_judge.py`, and the resulting (lower, honest)
accuracy is disclosed in the README rather than only reporting the
golden-dataset number.
"""

from src.golden_dataset import Conversation, Turn, QUALITY_GOOD, QUALITY_BORDERLINE, QUALITY_BAD


def _c(cid, agent_text, label):
    return Conversation(
        conversation_id=cid,
        turns=[Turn("user", "I have an issue with my account."), Turn("agent", agent_text)],
        gold_label=label,
        gold_reason="Hand-labeled paraphrase, held out from judge-marker design.",
    )


def generate_paraphrase_holdout():
    return [
        _c("hold-bad-1", "I don't really have access to that on my end, sorry.", QUALITY_BAD),
        _c("hold-bad-2", "You'll need to reach out to your account manager for billing stuff.", QUALITY_BAD),
        _c("hold-bad-3", "Could you power cycle the machine and see if that helps?", QUALITY_BAD),
        _c("hold-bad-4", "That's outside what I can help with, unfortunately.", QUALITY_BAD),
        _c("hold-good-1", "I've gone ahead and corrected the charge — ref #48213, should land in 2 days.", QUALITY_GOOD),
        _c("hold-good-2", "Fixed on our end and confirmed working — I also opened ticket #778 so the team patches the underlying bug.", QUALITY_GOOD),
        _c("hold-borderline-1", "Try clearing cookies and cache in your browser settings.", QUALITY_BORDERLINE),
        _c("hold-borderline-2", "We're aware of similar cases and the team is investigating.", QUALITY_BORDERLINE),
    ]
