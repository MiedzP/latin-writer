"""Calibrate the Latin judge with labelled examples, and measure whether it helps.

"Training" the judge here means showing it labelled examples of what counts as an error, in the
prompt (few-shot). Nothing is fine-tuned.

Setup:
  - example pool: batch caesar1 (keys say which items have a planted error and what changed);
  - held-out test: batches cicero1 and poetry1. These never appear as examples.
  - variants: zero-shot (no examples) and few-shot (k labelled examples from the pool).

The keys are planted-error labels, so they are a proxy for human judgment. Once you have judged
a batch, `--gold-from` can use your answers for the test items instead.

Run (needs GROQ_API_KEY and LATINWRITER_PROVIDER=groq, or ANTHROPIC_API_KEY):
  python -m evals.judge_calibrate --k 6
"""
import argparse
import json
import random

from latinwriter.config import JUDGE_MODEL
from latinwriter.judge import JUDGE_SYSTEM
from latinwriter.llm import ask, extract_json

from .common import RESULTS, save

EXAMPLE_BATCH = "caesar1"
FAILURES: list[str] = []
TEST_BATCHES = ["cicero1", "poetry1"]


def _items(batch: str) -> list[dict]:
    texts = json.loads((RESULTS / f"batch_{batch}.json").read_text(encoding="utf-8"))
    key = json.loads((RESULTS / f"key_{batch}.json").read_text(encoding="utf-8"))
    out = []
    for it in texts:
        truth = key[it["id"]]
        out.append({"id": it["id"], "batch": batch, "text": it["text"],
                    "is_error": truth["label"] == "error", "change": truth.get("change"),
                    "form": "prose" if truth.get("kind") == "prose" else "hexameter"})
    return out


def _example_block(examples: list[dict]) -> str:
    lines = ["Labelled examples (use them to calibrate what counts as an error; they are not the text you judge):"]
    for ex in examples:
        verdict = "ERROR" if ex["is_error"] else "CORRECT"
        why = f" (changed: {ex['change']})" if ex["is_error"] else ""
        lines.append(f'- "{ex["text"]}" -> {verdict}{why}')
    return "\n".join(lines)


def _judge_flags(text: str, form: str, examples: list[dict]) -> tuple[bool, dict]:
    user = f"Form: {form}. Requested style: classical Latin.\n\nLatin text:\n{text}"
    system = JUDGE_SYSTEM
    if examples:
        user = _example_block(examples) + "\n\n" + user
    last = None
    for _ in range(3):  # malformed JSON is retried; a judgment that never parses is counted, not hidden
        try:
            reply = ask([{"role": "user", "content": user}], system, model=JUDGE_MODEL, max_tokens=2000, json_mode=True)
            result = extract_json(reply)
            flagged = bool(result.get("errors")) or int(result.get("score", 100)) < 60
            return flagged, result
        except Exception as err:
            if "Failed to generate JSON" not in str(err) and not isinstance(err, (ValueError, KeyError)):
                raise
            last = err
    FAILURES.append(str(last)[:120])
    return False, {"unparseable": True}


def _few_shot_examples(pool: list[dict], k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    errors = [it for it in pool if it["is_error"]]
    clean = [it for it in pool if not it["is_error"]]
    rng.shuffle(errors)
    rng.shuffle(clean)
    half = k // 2
    return errors[:half] + clean[: k - half]


def _metrics(rows: list[dict]) -> dict:
    tp = sum(1 for r in rows if r["is_error"] and r["flagged"])
    fn = sum(1 for r in rows if r["is_error"] and not r["flagged"])
    fp = sum(1 for r in rows if not r["is_error"] and r["flagged"])
    tn = sum(1 for r in rows if not r["is_error"] and not r["flagged"])
    total = tp + fn + fp + tn
    return {
        "n": total,
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "error_recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "clean_pass_rate": round(tn / (tn + fp), 3) if tn + fp else None,
    }


def run(k: int = 6, seed: int = 0, limit: int | None = None) -> dict:
    pool = _items(EXAMPLE_BATCH)
    examples = _few_shot_examples(pool, k, seed)
    example_ids = {e["id"] for e in examples}
    test = [it for b in TEST_BATCHES for it in _items(b)]
    if limit:
        test = test[:limit]

    variants = {"zero_shot": [], "few_shot": []}
    for it in test:
        for name, exs in (("zero_shot", []), ("few_shot", examples)):
            flagged, _ = _judge_flags(it["text"], it["form"], exs)
            variants[name].append(dict(it, flagged=flagged))
        print(f"{it['batch']} {it['id']}: truth={'error' if it['is_error'] else 'clean'}")

    report = {
        "eval": "judge_calibration",
        "example_batch": EXAMPLE_BATCH,
        "example_ids": sorted(example_ids),
        "test_batches": TEST_BATCHES,
        "k": k,
        "zero_shot": _metrics(variants["zero_shot"]),
        "few_shot": _metrics(variants["few_shot"]),
        "unparseable_judgments": len(FAILURES),
    }
    save(f"judge_calibration_k{k}.json", report)
    print(json.dumps({"zero_shot": report["zero_shot"], "few_shot": report["few_shot"]}, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(k=a.k, seed=a.seed, limit=a.limit)
