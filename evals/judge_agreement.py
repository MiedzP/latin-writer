"""Measure the automated judge against human answers on the holdout (never shown to the judge).

Two variants on the same items:
  - no_examples: the judge with no human-labelled examples;
  - with_examples: the judge with data/judge_examples.json loaded (what the pipeline uses).

Gold = the human's answer. The judge "flags" an item when it reports any error or scores it below 60.
Unparseable judge replies are counted separately and left out of the agreement numbers.

Run:  python -m evals.judge_agreement
"""
import json

from latinwriter.config import JUDGE_MODEL, ROOT
from latinwriter.judge import _ask_judge

from .common import save


def _flag(result: dict) -> bool | None:
    if any("not valid JSON" in str(e.get("problem", "")) for e in result.get("errors", [])):
        return None  # unparseable: no verdict
    return bool(result.get("errors")) or int(result.get("score", 100)) < 60


def _metrics(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["flagged"] is not None]
    tp = sum(1 for r in scored if r["gold_error"] and r["flagged"])
    fn = sum(1 for r in scored if r["gold_error"] and not r["flagged"])
    fp = sum(1 for r in scored if not r["gold_error"] and r["flagged"])
    tn = sum(1 for r in scored if not r["gold_error"] and not r["flagged"])
    n = len(scored)
    return {
        "scored": n,
        "unparseable": len(rows) - n,
        "agreement": round((tp + tn) / n, 3) if n else None,
        "error_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "clean_pass_rate": round(tn / (tn + fp), 3) if tn + fp else None,
    }


def run() -> dict:
    holdout = json.loads((ROOT / "data" / "judge_holdout.json").read_text(encoding="utf-8"))
    variants = {}
    for name, use_examples in (("no_examples", False), ("with_examples", True)):
        rows = []
        for item in holdout:
            result = _ask_judge(item["text"], "prose", "caesar", "x", None, JUDGE_MODEL, use_examples)
            rows.append({"id": item["id"], "gold_error": item["is_error"], "flagged": _flag(result),
                         "score": result.get("score")})
            print(f"{name:<14} {item['id']}: gold={'error' if item['is_error'] else 'clean'} "
                  f"judge={'error' if _flag(result) else 'clean' if _flag(result) is False else 'unparseable'}")
        variants[name] = _metrics(rows)
    report = {"eval": "judge_agreement", "holdout_items": len(holdout),
              "holdout_errors": sum(x["is_error"] for x in holdout), **variants}
    save("judge_agreement.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
