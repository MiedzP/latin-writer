"""Verb eval layer: are verbs included, and are they explicit, implied, or missing?

Three commands:
  coverage  how often real held-out sentences with a verb are caught by the lexicon detector
            (also lists uncaught sentences, to show what the lexicon needs);
  make      build a blinded batch: clean sentences that have one lexicon verb, and the same
            sentences with that verb deleted;
  judge     three-way human judgment for each item:
              v = the verb is there (explicit)
              i = no verb written, but the sentence is still grammatical (implied, e.g. omitted "est")
              n = no verb and the sentence needs one (missing)
  score     detector against human judgments, and the human split for deleted items.

Usage (run in your own terminal for judge):
  python -m evals.verb_eval coverage
  python -m evals.verb_eval make --n 40 --batch verbs1
  python -m evals.verb_eval judge --batch verbs1
  python -m evals.verb_eval score --batch verbs1
"""
import argparse
import json
import random

from .common import RESULTS, held_out
from .verbs import LEXICON, find_verbs, has_explicit_verb, remove_verb

ANSWERS = {"v": "explicit", "i": "implied", "n": "missing"}


def _sentences(author: str | None = None) -> list[dict]:
    rows = [r for k in ("prose", "verse") for r in held_out(k)]
    rows = [r for r in rows if r["text"].rstrip()[-1:] in ".;!?" and 5 <= len(r["text"].split()) <= 20]
    if author:
        rows = [r for r in rows if r["author"] == author]
    return rows


def coverage() -> dict:
    rows = _sentences()
    found = [r for r in rows if has_explicit_verb(r["text"])]
    missed = [r for r in rows if not has_explicit_verb(r["text"])]
    report = {
        "eval": "verb_lexicon_coverage",
        "sentences": len(rows),
        "with_lexicon_verb": len(found),
        "coverage": round(len(found) / len(rows), 3) if rows else None,
        "lexicon_size": len(LEXICON),
        "sample_without_verb": [{"author": r["author"], "text": r["text"]} for r in missed[:40]],
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "verb_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"sentences {len(rows)}, with a lexicon verb {len(found)} (coverage {report['coverage']})")
    print("(sentences without a verb are not all errors: many are real fragments or use verbs outside the lexicon)")
    return report


def make(n: int, batch: str, seed: int = 0, author: str | None = None) -> None:
    rng = random.Random(seed)
    pool = [r for r in _sentences(author) if len(find_verbs(r["text"])) == 1]
    rng.shuffle(pool)
    items, key = [], {}
    for r in pool:
        if len(items) >= n:
            break
        if rng.random() < 0.5:
            text, label, removed = r["text"], "clean", None
        else:
            res = remove_verb(r["text"], 0)
            if res is None:
                continue
            text, removed = res
            label = "deleted"
        item_id = f"{batch}-{len(items):03d}"
        items.append({"id": item_id, "text": text})
        key[item_id] = {"label": label, "removed": removed, "author": r["author"], "original": r["text"]}
    rng.shuffle(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"verbbatch_{batch}.json").write_text(json.dumps(items, indent=2, ensure_ascii=False), encoding="utf-8")
    (RESULTS / f"verbkey_{batch}.json").write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"batch {batch}: {len(items)} items")


def judge(batch: str) -> None:
    items = json.loads((RESULTS / f"verbbatch_{batch}.json").read_text(encoding="utf-8"))
    out_path = RESULTS / f"verbjudg_{batch}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(todo)} items to judge. v = verb is there, i = implied (no verb, still grammatical), "
          f"n = missing (needs a verb), s = skip, q = quit")
    with out_path.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            print(f"\n[{k}/{len(todo)}] {it['text']}")
            while True:
                ans = input("verb? [v/i/n/s/q] ").strip().lower()
                if ans in ("v", "i", "n", "s", "q"):
                    break
            if ans == "q":
                break
            if ans == "s":
                continue
            note = input("note (optional): ").strip()
            f.write(json.dumps({"id": it["id"], "answer": ANSWERS[ans], "note": note}, ensure_ascii=False) + "\n")
            f.flush()
    print(f"saved to {out_path}")


def score(batch: str) -> dict:
    items = {it["id"]: it for it in json.loads((RESULTS / f"verbbatch_{batch}.json").read_text(encoding="utf-8"))}
    key = json.loads((RESULTS / f"verbkey_{batch}.json").read_text(encoding="utf-8"))
    judged = [json.loads(l) for l in (RESULTS / f"verbjudg_{batch}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    by_label: dict[str, dict[str, int]] = {"clean": {}, "deleted": {}}
    detector_agree = detector_total = 0
    surprises = []
    for j in judged:
        truth = key[j["id"]]
        label = truth["label"]
        by_label[label][j["answer"]] = by_label[label].get(j["answer"], 0) + 1
        # detector on the shown text: should say "explicit" exactly when the human says explicit
        det_explicit = has_explicit_verb(items[j["id"]]["text"])
        human_explicit = j["answer"] == "explicit"
        detector_total += 1
        detector_agree += det_explicit == human_explicit
        if det_explicit != human_explicit:
            surprises.append({"id": j["id"], "label": label, "human": j["answer"],
                              "detector_explicit": det_explicit, "text": items[j["id"]]["text"],
                              "note": j["note"]})

    report = {
        "batch": batch,
        "judged": len(judged),
        "human_answers_on_clean_sentences": by_label["clean"],
        "human_answers_on_deleted_verbs": by_label["deleted"],
        "detector_human_agreement": round(detector_agree / detector_total, 3) if detector_total else None,
        "detector_disagreements": surprises,
    }
    (RESULTS / f"verb_score_{batch}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"judged {len(judged)}")
    print(f"clean sentences:  {by_label['clean']}")
    print(f"deleted verbs:    {by_label['deleted']}")
    print(f"detector vs human (explicit or not): {report['detector_human_agreement']}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("coverage")
    m = sub.add_parser("make")
    m.add_argument("--n", type=int, default=40)
    m.add_argument("--batch", required=True)
    m.add_argument("--author", default=None)
    j = sub.add_parser("judge")
    j.add_argument("--batch", required=True)
    s = sub.add_parser("score")
    s.add_argument("--batch", required=True)
    a = ap.parse_args()
    if a.cmd == "coverage":
        coverage()
    elif a.cmd == "make":
        make(a.n, a.batch, author=a.author)
    elif a.cmd == "judge":
        judge(a.batch)
    else:
        score(a.batch)
