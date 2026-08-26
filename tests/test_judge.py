import pytest

from src.golden_dataset import (
    generate_golden_dataset, QUALITY_GOOD, QUALITY_BORDERLINE, QUALITY_BAD,
)
from src.paraphrase_holdout import generate_paraphrase_holdout
from src.judge import HeuristicJudge, PromptedLLMJudge
from src.evaluate import evaluate_judge


class TestGoldenDataset:
    def test_generates_expected_size_and_tier_balance(self):
        convos = generate_golden_dataset(n_per_tier=20)
        assert len(convos) == 60
        labels = [c.gold_label for c in convos]
        assert labels.count(QUALITY_GOOD) == 20
        assert labels.count(QUALITY_BORDERLINE) == 20
        assert labels.count(QUALITY_BAD) == 20

    def test_every_conversation_has_a_gold_label_and_reason(self):
        convos = generate_golden_dataset()
        for c in convos:
            assert c.gold_label in (QUALITY_GOOD, QUALITY_BORDERLINE, QUALITY_BAD)
            assert c.gold_reason and len(c.gold_reason) > 10

    def test_deterministic_with_fixed_seed(self):
        a = generate_golden_dataset(seed=7)
        b = generate_golden_dataset(seed=7)
        assert [c.gold_label for c in a] == [c.gold_label for c in b]

    def test_transcript_includes_both_speakers(self):
        convo = generate_golden_dataset()[0]
        t = convo.transcript()
        assert "user:" in t
        assert "agent:" in t


class TestHeuristicJudgeOnGoldenSet:
    def test_agreement_rate_on_golden_set_is_high(self):
        """
        The judge should closely track the golden-dataset labels it was
        tuned against. This alone is NOT proof of a good judge — see
        test_generalization_to_paraphrases_is_the_real_test below — but a
        judge that can't even match the set it was built against would be
        a genuine bug.
        """
        results = evaluate_judge(HeuristicJudge())
        assert results["agreement_rate"] >= 0.95

    def test_no_disagreements_are_silently_dropped(self):
        """Every disagreement must be individually inspectable, not just counted."""
        results = evaluate_judge(HeuristicJudge())
        assert len(results["disagreements"]) == results["total"] - results["correct"]
        for d in results["disagreements"]:
            assert d["judge_reason"]
            assert d["gold_reason"]


class TestGeneralization:
    def test_generalization_to_paraphrases_is_the_real_test(self):
        """
        This is the test that matters most in this file.

        The first version of HeuristicJudge scored 100% on the golden
        dataset — which, following this portfolio's standing rule that a
        suspiciously perfect result is a red flag to investigate rather
        than a result to report, was checked against hand-written
        paraphrases using different wording than the golden-set templates.

        It generalizes badly: agreement drops to 25% (2/8) on paraphrased
        text that a human would judge identically to the golden-set
        cases. This test locks that honest, low number in as a known,
        disclosed limitation of a keyword-based judge — it must NOT
        silently improve to something that looks like accidental
        overfitting to this specific holdout either; if someone tunes the
        markers against these exact holdout phrases without a THIRD
        independent check, that's the same mistake happening again.
        """
        results = evaluate_judge(HeuristicJudge(), generate_paraphrase_holdout())
        # Documents the real, measured generalization gap. This is
        # intentionally a narrow band, not just a floor: a HeuristicJudge
        # scoring much higher here likely means the holdout examples
        # leaked into the marker list rather than the judge having
        # improved genuinely.
        assert 0.15 <= results["agreement_rate"] <= 0.45

    def test_golden_set_score_is_not_evidence_of_generalization(self):
        """
        Explicit regression guard against the exact mistake this project
        found and fixed once already: reporting only the golden-set
        number as if it were the whole story.
        """
        golden_results = evaluate_judge(HeuristicJudge())
        holdout_results = evaluate_judge(HeuristicJudge(), generate_paraphrase_holdout())
        gap = golden_results["agreement_rate"] - holdout_results["agreement_rate"]
        assert gap > 0.4, (
            "Expected a large, disclosed gap between golden-set and "
            "holdout agreement, demonstrating why golden-set accuracy "
            "alone is not sufficient evidence a judge generalizes."
        )


class TestPromptedLLMJudgeIsHonestlyUnimplemented:
    def test_raises_not_implemented_with_clear_reason(self):
        judge = PromptedLLMJudge()
        convo = generate_golden_dataset()[0]
        with pytest.raises(NotImplementedError, match="no API key"):
            judge.judge(convo)

    def test_docstring_documents_the_real_evaluation_prompt(self):
        assert "LLM-as-Judge" in PromptedLLMJudge.__doc__
        assert "good" in PromptedLLMJudge.__doc__
        assert "borderline" in PromptedLLMJudge.__doc__
        assert "bad" in PromptedLLMJudge.__doc__
