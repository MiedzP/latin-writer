"""Human judging for the eval loop, with planted errors so the answers are known.

Each batch mixes two kinds of Latin:
  - clean: real held-out lines or sentences from the corpus (grammatical);
  - planted: the same kind of text with one word ending changed, so it is ungrammatical.

The judge (you) sees the text with no labels and answers "is this grammatical Latin?".
The answer key is kept separately in evals/results/ (git-ignored) and is only read by `score`.

Usage (run in your own terminal, since it reads from the keyboard):
  python -m evals.human_judge make --n 30 --batch b1
  python -m evals.human_judge judge --batch b1
  python -m evals.human_judge score --batch b1
Keys: y = grammatical, n = has an error, s = skip, q = quit and save.
"""
import argparse
import json
import random
import re
from pathlib import Path

from .common import RESULTS, clean, held_out

# Ending swaps that usually make a Latin word wrong in its case, number or tense.
SWAPS = [("orum", "ae"), ("ibus", "os"), ("um", "us"), ("us", "um"), ("am", "as"), ("as", "am"),
         ("is", "em"), ("em", "is"), ("ae", "am"), ("it", "ant"), ("unt", "it")]


# Finite-verb ending swaps (person/number/tense changes). Used by the verb-focused batch.
VERB_SWAPS = [("it", "et"), ("et", "it"), ("at", "it"), ("it", "at"), ("unt", "it"), ("it", "unt"),
              ("ant", "at"), ("at", "ant"), ("nt", "t"), ("t", "nt")]
VERB_SUFFIX = ("it", "et", "at", "unt", "ant", "t", "nt")


def plant_error(text: str, rng: random.Random, swaps=SWAPS) -> tuple[str, str] | None:
    """Return (corrupted_text, changed_word) or None if no word can be changed."""
    words = text.split()
    candidates = []
    for i, w in enumerate(words):
        core = re.sub(r"[^A-Za-z]+$", "", w)
        for old, new in swaps:
            if len(core) >= 4 and core.lower().endswith(old):
                candidates.append((i, core, old, new))
    if not candidates:
        return None
    i, core, old, new = rng.choice(candidates)
    suffix = words[i][len(core):]
    new_core = core[: len(core) - len(old)] + new
    words[i] = new_core + suffix
    return " ".join(words), core + " -> " + new_core


def make(n: int, batch: str, seed: int = 0, focus: str | None = None,
         author: str | None = None, kind: str | None = None) -> None:
    """Build a blinded batch.

    focus=None:      mixed short clips (legacy batch b1).
    focus='verb':    complete sentences, planted errors only in verb endings.
    focus='grammar': complete sentences, planted errors in noun/adjective endings and verb endings.
    author / kind:   restrict the held-out pool (e.g. author='Caesar', kind='prose').
    """
    rng = random.Random(seed)
    kinds = [kind] if kind else ["verse", "prose"]
    base = [r for k in kinds for r in held_out(k)]
    if author:
        base = [r for r in base if r["author"] == author]
    if focus in ("verb", "grammar"):
        pool = [r for r in base
                if 6 <= len(r["text"].split()) <= 20 and r["text"].rstrip()[-1:] in ".;!?"]
    else:
        pool = [r for r in base if 6 <= len(r["text"].split()) <= 16]
    rng.shuffle(pool)
    items, key = [], {}
    for r in pool:
        if len(items) >= n:
            break
        planted = rng.random() < 0.5
        category = None
        if planted:
            if focus == "verb":
                swaps, category = VERB_SWAPS, "verb"
            elif focus == "grammar":
                # mix: about half nominal endings, half verb endings
                if rng.random() < 0.5:
                    swaps, category = VERB_SWAPS, "verb"
                else:
                    swaps, category = SWAPS, "nominal"
            else:
                swaps = SWAPS
            res = plant_error(r["text"], rng, swaps)
            if res is None:
                continue
            text, change = res
            label = "error"
        else:
            text, change, label = r["text"], None, "clean"
        item_id = f"{batch}-{len(items):03d}"
        items.append({"id": item_id, "text": text})
        key[item_id] = {"label": label, "change": change, "category": category,
                        "author": r["author"], "kind": r["kind"]}
    rng.shuffle(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"batch_{batch}.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"key_{batch}.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"batch {batch}: {len(items)} items -> {RESULTS / f'batch_{batch}.json'}")


def judge(batch: str) -> None:
    items = json.loads((RESULTS / f"batch_{batch}.json").read_text(encoding="utf-8"))
    out_path = RESULTS / f"judgments_{batch}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(todo)} items to judge ({len(done)} already done). y = grammatical, n = has an error, s = skip, q = quit")
    with out_path.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            print(f"\n[{k}/{len(todo)}] {it['text']}")
            while True:
                ans = input("grammatical? [y/n/s/q] ").strip().lower()
                if ans in ("y", "n", "s", "q"):
                    break
            if ans == "q":
                break
            if ans == "s":
                continue
            note = input("note (optional, Enter to skip): ").strip()
            f.write(json.dumps({"id": it["id"], "says_error": ans == "n", "note": note}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"saved to {out_path}")


def score(batch: str) -> dict:
    key = json.loads((RESULTS / f"key_{batch}.json").read_text(encoding="utf-8"))
    judged = [json.loads(l) for l in (RESULTS / f"judgments_{batch}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    tp = fp = tn = fn = 0
    misses, false_alarms = [], []
    by_cat: dict[str, dict[str, int]] = {}
    for j in judged:
        truth = key[j["id"]]
        is_error = truth["label"] == "error"
        cat = truth.get("category") or ("clean" if not is_error else "unlabelled")
        bucket = by_cat.setdefault(cat, {"caught": 0, "missed": 0, "clean_ok": 0, "clean_flagged": 0})
        if is_error and j["says_error"]:
            tp += 1
            bucket["caught"] += 1
        elif is_error:
            fn += 1
            bucket["missed"] += 1
            misses.append({"id": j["id"], "change": truth["change"], "category": cat, "note": j["note"]})
        elif j["says_error"]:
            fp += 1
            bucket["clean_flagged"] += 1
            false_alarms.append({"id": j["id"], "author": truth["author"], "note": j["note"]})
        else:
            tn += 1
            bucket["clean_ok"] += 1
    total = tp + fp + tn + fn
    report = {
        "batch": batch, "judged": total,
        "planted_caught": tp, "planted_missed": fn,
        "clean_flagged": fp, "clean_passed": tn,
        "accuracy": round((tp + tn) / total, 3) if total else None,
        "by_category": by_cat,
        "missed_planted": misses, "false_alarms": false_alarms,
    }
    (RESULTS / f"human_score_{batch}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"judged {total}: caught {tp}/{tp + fn} planted errors, flagged {fp}/{fp + tn} clean texts, accuracy {report['accuracy']}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, default=30)
    m.add_argument("--batch", required=True)
    m.add_argument("--focus", choices=["verb", "grammar"], default=None)
    m.add_argument("--author", default=None, help="e.g. Caesar, Cicero, Vergil, Ovid")
    m.add_argument("--kind", choices=["verse", "prose"], default=None)
    j = sub.add_parser("judge")
    j.add_argument("--batch", required=True)
    s = sub.add_parser("score")
    s.add_argument("--batch", required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.n, a.batch, focus=a.focus, author=a.author, kind=a.kind)
    elif a.cmd == "judge":
        judge(a.batch)
    else:
        score(a.batch)
