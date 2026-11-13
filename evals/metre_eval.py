"""Evaluate the metre checker on held-out real text, sweeping its violation tolerance.

Positives: held-out verse lines of Vergil's Aeneid and Ovid's Metamorphoses (dactylic hexameters).
Negatives: held-out prose sentences from Caesar and Cicero.

The checker can tolerate a number of heavy-syllables-in-short-slots (max_violations). A higher
tolerance finds more real verse but also accepts more prose. The sweep shows the trade-off, and
the default in latinwriter.meter should be the setting that keeps false positives low.

Run:  python -m evals.metre_eval
"""
from latinwriter.meter import scan

from .common import clean, held_out, save

SETTINGS = [0, 1, 2, 3]


def evaluate(max_violations: int, verse: list[dict], prose: list[dict]) -> dict:
    tp = sum(1 for r in verse if scan(r["text"], "hexameter", max_violations).ok)
    fn = len(verse) - tp
    fp = sum(1 for r in prose if scan(r["text"], "hexameter", max_violations).ok)
    tn = len(prose) - fp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "max_violations": max_violations,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": round((tp + tn) / (len(verse) + len(prose)), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "false_positive_rate": round(fp / len(prose), 4) if prose else 0.0,
        "f1": round(f1, 4),
    }


def run() -> dict:
    verse = [dict(r, text=clean(r["text"])) for r in held_out("verse") if r["author"] in ("Vergil", "Ovid")]
    prose = [dict(r, text=clean(r["text"])) for r in held_out("prose") if r["author"] in ("Caesar", "Cicero")]

    sweep = [evaluate(v, verse, prose) for v in SETTINGS]
    print(f"verse lines {len(verse)}, prose sentences {len(prose)}")
    print(f"{'max_viol':>8} {'recall':>7} {'FP rate':>8} {'precision':>10} {'accuracy':>9} {'F1':>6}")
    for row in sweep:
        print(f"{row['max_violations']:>8} {row['recall']:>7.3f} {row['false_positive_rate']:>8.3f} "
              f"{row['precision']:>10.3f} {row['accuracy']:>9.3f} {row['f1']:>6.3f}")

    # Examples at the strictest setting, to see what is being missed and what is wrongly accepted.
    examples = {
        "missed_verse_at_0": [
            {"author": r["author"], "text": r["text"], "reason": scan(r["text"], "hexameter", 0).reason}
            for r in verse if not scan(r["text"], "hexameter", 0).ok][:25],
        "false_positive_prose_at_0": [
            {"author": r["author"], "text": r["text"]}
            for r in prose if scan(r["text"], "hexameter", 0).ok][:25],
    }
    report = {"eval": "metre_hexameter", "sweep": sweep, "examples": examples,
              "positives": len(verse), "negatives": len(prose)}
    save("metre_eval.json", report)
    return report


if __name__ == "__main__":
    run()
