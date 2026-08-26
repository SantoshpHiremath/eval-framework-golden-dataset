"""
Sanity checks on the golden dataset's own annotation quality — the
project's version of an annotator spot-checking their own labels for
consistency, since a golden dataset is only as trustworthy as its labels.
"""

from src.golden_dataset import (
    generate_golden_dataset, GOOD_RESPONSES, BORDERLINE_RESPONSES, BAD_RESPONSES,
)


def test_response_pools_do_not_overlap():
    """A response template appearing in two quality tiers would make the
    golden labels internally inconsistent."""
    good, borderline, bad = set(GOOD_RESPONSES), set(BORDERLINE_RESPONSES), set(BAD_RESPONSES)
    assert not (good & borderline)
    assert not (good & bad)
    assert not (borderline & bad)


def test_each_conversation_has_exactly_one_user_and_one_agent_turn():
    for c in generate_golden_dataset():
        speakers = [t.speaker for t in c.turns]
        assert speakers == ["user", "agent"]


def test_conversation_ids_are_unique():
    convos = generate_golden_dataset()
    ids = [c.conversation_id for c in convos]
    assert len(ids) == len(set(ids))


def test_gold_reasons_are_tier_specific_not_copy_pasted():
    """The reasoning text should genuinely differ by quality tier, not be
    a single generic string applied to every label."""
    convos = generate_golden_dataset()
    reasons_by_label = {}
    for c in convos:
        reasons_by_label.setdefault(c.gold_label, set()).add(c.gold_reason)
    # Each tier should have its own distinct reasoning text.
    all_reasons = [r for reasons in reasons_by_label.values() for r in reasons]
    assert len(set(all_reasons)) == len(reasons_by_label)
