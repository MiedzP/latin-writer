"""Turn a human-judged batch into calibration examples for the automated judge.

Gold label = the human's answer (says_error), not the planted key. Where they disagree (for example a
planted change that is still valid Latin), the human wins, and the key is kept only for reference.

Split:
  - holdout: N items, never shown to the judge, used only by evals/judge_agreement.py;
  - examples: the rest, balanced; K of them are written to data/judge_examples.json and loaded
    automatically by latinwriter.judge on every call.

Run:  python -m evals.build_judge_examples --batch caesar1 --holdout 20 --k 8
"""
import argparse
import json
import random

from latinwriter.config import ROOT

from .common import RESULTS

DATA = ROOT / "data"


def build(batch: str, holdout_n: int, k: int, seed: int = 0) -> dict:
    texts = {it["id"]: it["text"] for it in json.loads((RESULTS / f"batch_{batch}.json").read_text(encoding="utf-8"))}
    key = json.loads((RESULTS / f"key_{batch}.json").read_text(encoding="utf-8"))
    judged = [json.loads(l) for l in (RESULTS / f"judgments_{batch}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    labelled = [{"id": j["id"], "text": texts[j["id"]], "is_error": bool(j["says_error"]),
                 "note": j.get("note", ""), "planted": key[j["id"]]["label"] == "error",
                 "change": key[j["id"]].get("change")} for j in judged]

    rng = random.Random(seed)
    rng.shuffle(labelled)
    holdout = labelled[:holdout_n]
    pool = labelled[holdout_n:]

    errors = [x for x in pool if x["is_error"]]
    clean = [x for x in pool if not x["is_error"]]
    rng.shuffle(errors)
    rng.shuffle(clean)
    chosen = errors[: k // 2] + clean[: k - k // 2]
    rng.shuffle(chosen)

    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "judge_examples.json").write_text(json.dumps(
        [{"text": x["text"], "is_error": x["is_error"], "note": x["note"], "change": x["change"]} for x in chosen],
        indent=2, ensure_ascii=False), encoding="utf-8")
    (DATA / "judge_holdout.json").write_text(json.dumps(holdout, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = {
        "batch": batch,
        "labelled": len(labelled),
        "human_errors": sum(x["is_error"] for x in labelled),
        "examples_used": len(chosen),
        "examples_errors": sum(x["is_error"] for x in chosen),
        "holdout": len(holdout),
        "holdout_errors": sum(x["is_error"] for x in holdout),
        "human_vs_planted_disagreements": sum(x["is_error"] != x["planted"] for x in labelled),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--holdout", type=int, default=20)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.batch, a.holdout, a.k, a.seed)
