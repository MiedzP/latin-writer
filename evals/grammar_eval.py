"""Grammar eval layers built from the Latin grammar book (data/grammar, local only).

The book's usage examples are authentic Latin sentences that illustrate a rule, so they are used as
clean (grammatical) items. Planted errors are made with the same swap rules as the human judge,
and each item is tagged with the book's part (verbal system, syntax of cases, syntax of clauses,
pronouns, nouns and adjectives), so results can be read per grammatical area.

The batch and key use the same file layout as evals/human_judge.py, so the existing commands work:
  python -m evals.grammar_eval make --n 60 --batch g1
  python -m evals.human_judge judge --batch g1
  python -m evals.human_judge score --batch g1
  python -m evals.grammar_eval report --batch g1

`coverage` checks that the lexicon verb detector recognises the verbs in the book's verbal examples.
"""
import argparse
import json
import random

from latinwriter.grammar import load_examples

from .common import RESULTS
from .human_judge import SWAPS, VERB_SWAPS, plant_error
from .verbs import has_explicit_verb

AREAS = ["verbal system", "syntax of cases", "syntax of clauses", "pronouns", "nouns and adjectives"]


def _pool(areas: list[str]) -> list[dict]:
    rows = []
    for area in areas:
        rows += [e for e in load_examples(area) if 3 <= len(e["latin"].split()) <= 14]
    return rows


def make(n: int, batch: str, seed: int = 0, areas: list[str] | None = None) -> None:
    rng = random.Random(seed)
    pool = _pool(areas or AREAS)
    rng.shuffle(pool)
    items, key = [], {}
    want_error = True
    for e in pool:
        if len(items) >= n:
            break
        if want_error:
            swaps = VERB_SWAPS if e["part"] == "verbal system" else SWAPS
            res = plant_error(e["latin"] + ".", rng, swaps, min_len=3)
            if res is None:
                continue  # nothing to change in this sentence; try the next one
            text, change = res
            label = "error"
        else:
            text, change, label = e["latin"] + ".", None, "clean"
        want_error = not want_error  # alternate, so the batch is balanced
        item_id = f"{batch}-{len(items):03d}"
        items.append({"id": item_id, "text": text})
        key[item_id] = {"label": label, "change": change, "category": e["part"],
                        "author": e["cite"] or "grammar book", "kind": "gold", "page": e["page"]}
    rng.shuffle(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"batch_{batch}.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"key_{batch}.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"batch {batch}: {len(items)} items from {len(pool)} book examples")


def report(batch: str) -> dict:
    path = RESULTS / f"human_score_{batch}.json"
    score = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    print(f"{'area':<22} {'caught':>7} {'missed':>7} {'clean ok':>9} {'flagged':>8}")
    for area in AREAS:
        b = score["by_category"].get(area, {"caught": 0, "missed": 0, "clean_ok": 0, "clean_flagged": 0})
        planted = b["caught"] + b["missed"]
        clean = b["clean_ok"] + b["clean_flagged"]
        rows.append({"area": area, **b,
                     "error_recall": round(b["caught"] / planted, 3) if planted else None,
                     "clean_pass_rate": round(b["clean_ok"] / clean, 3) if clean else None})
        print(f"{area:<22} {b['caught']:>3}/{planted:<3} {b['missed']:>7} {b['clean_ok']:>9} {b['clean_flagged']:>8}")
    out = {"batch": batch, "areas": rows, "overall_accuracy": score.get("accuracy")}
    (RESULTS / f"grammar_report_{batch}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def coverage() -> dict:
    rows = _pool(AREAS)
    verbal = [e for e in rows if e["part"] == "verbal system"]
    found = [e for e in verbal if has_explicit_verb(e["latin"])]
    report = {"verbal_examples": len(verbal), "with_lexicon_verb": len(found),
              "coverage": round(len(found) / len(verbal), 3) if verbal else None}
    print(report)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, default=60)
    m.add_argument("--batch", required=True)
    r = sub.add_parser("report")
    r.add_argument("--batch", required=True)
    sub.add_parser("coverage")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.n, a.batch)
    elif a.cmd == "report":
        report(a.batch)
    else:
        coverage()
