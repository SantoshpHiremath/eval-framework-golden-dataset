"""
Judge-vs-golden-label evaluation harness.

This is the loop of comparing automated evaluation scores with human
feedback. It runs a Judge over every conversation in
the golden dataset, compares its verdict to the human-style gold label,
and reports:

  - overall agreement rate (exact label match)
  - a confusion matrix (gold label x judge label)
  - the actual disagreement cases, with both the judge's reasoning and
    the gold reasoning shown side by side, so a real reviewer could look
    at exactly where and why the judge is wrong — not just a single
    summary number.

No result here is hidden or smoothed over: if the judge disagrees with
the golden label, that case shows up in `disagreements`.
"""

from collections import Counter

from src.golden_dataset import generate_golden_dataset


def evaluate_judge(judge, conversations=None):
    if conversations is None:
        conversations = generate_golden_dataset()

    total = len(conversations)
    correct = 0
    confusion = Counter()  # (gold_label, judge_label) -> count
    disagreements = []

    for convo in conversations:
        judge_label, judge_reason = judge.judge(convo)
        confusion[(convo.gold_label, judge_label)] += 1

        if judge_label == convo.gold_label:
            correct += 1
        else:
            disagreements.append({
                "conversation_id": convo.conversation_id,
                "transcript": convo.transcript(),
                "gold_label": convo.gold_label,
                "gold_reason": convo.gold_reason,
                "judge_label": judge_label,
                "judge_reason": judge_reason,
            })

    agreement_rate = correct / total if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "agreement_rate": agreement_rate,
        "confusion_matrix": dict(confusion),
        "disagreements": disagreements,
    }


def print_report(results):
    print(f"Total conversations evaluated: {results['total']}")
    print(f"Judge/gold-label agreement: {results['correct']}/{results['total']} "
          f"({results['agreement_rate']:.1%})")
    print()
    print("Confusion matrix (gold_label -> judge_label : count):")
    for (gold, pred), count in sorted(results["confusion_matrix"].items()):
        marker = "" if gold == pred else "  <-- disagreement"
        print(f"  {gold:>10} -> {pred:<10} : {count}{marker}")
    print()
    print(f"Disagreement cases ({len(results['disagreements'])}):")
    for d in results["disagreements"]:
        print(f"  [{d['conversation_id']}] gold={d['gold_label']!r} "
              f"judge={d['judge_label']!r}")
        print(f"    gold reason:  {d['gold_reason']}")
        print(f"    judge reason: {d['judge_reason']}")


if __name__ == "__main__":
    from src.judge import HeuristicJudge

    judge = HeuristicJudge()
    results = evaluate_judge(judge)
    print_report(results)
