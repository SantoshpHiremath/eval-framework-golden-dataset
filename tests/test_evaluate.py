import pytest

from src.golden_dataset import Conversation, Turn, QUALITY_GOOD, QUALITY_BAD
from src.judge import Judge
from src.evaluate import evaluate_judge


class _AlwaysGoodJudge(Judge):
    """A deliberately wrong judge, used to test the evaluation harness itself."""
    def judge(self, conversation):
        return QUALITY_GOOD, "always good"


class _PerfectJudge(Judge):
    """A judge that just reads the gold label — used to test the harness reports 100% correctly."""
    def judge(self, conversation):
        return conversation.gold_label, "matches gold by construction"


def _mini_dataset():
    return [
        Conversation("a", [Turn("user", "x"), Turn("agent", "y")], gold_label=QUALITY_GOOD, gold_reason="r"),
        Conversation("b", [Turn("user", "x"), Turn("agent", "y")], gold_label=QUALITY_BAD, gold_reason="r"),
        Conversation("c", [Turn("user", "x"), Turn("agent", "y")], gold_label=QUALITY_BAD, gold_reason="r"),
    ]


def test_evaluate_judge_computes_correct_agreement_rate():
    results = evaluate_judge(_AlwaysGoodJudge(), _mini_dataset())
    assert results["total"] == 3
    assert results["correct"] == 1
    assert results["agreement_rate"] == pytest.approx(1 / 3)


def test_evaluate_judge_perfect_judge_has_no_disagreements():
    results = evaluate_judge(_PerfectJudge(), _mini_dataset())
    assert results["agreement_rate"] == 1.0
    assert results["disagreements"] == []


def test_evaluate_judge_disagreements_carry_both_reasonings():
    results = evaluate_judge(_AlwaysGoodJudge(), _mini_dataset())
    bad_case = next(d for d in results["disagreements"] if d["conversation_id"] == "b")
    assert bad_case["gold_label"] == QUALITY_BAD
    assert bad_case["judge_label"] == QUALITY_GOOD
    assert bad_case["judge_reason"] == "always good"


def test_confusion_matrix_counts_match_total():
    results = evaluate_judge(_AlwaysGoodJudge(), _mini_dataset())
    assert sum(results["confusion_matrix"].values()) == results["total"]


def test_evaluate_judge_handles_empty_dataset():
    results = evaluate_judge(_AlwaysGoodJudge(), [])
    assert results["total"] == 0
    assert results["agreement_rate"] == 0.0
    assert results["disagreements"] == []
