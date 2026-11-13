"""Generation eval: does the writing agent produce metrical Latin in the requested author's style?

Each case is (form, style, theme). Every case is run twice: with the author's exemplars shown to the
generator (training-split lines only) and without, so the effect of the exemplar bank can be read off.

Measures per run:
  - result: the agent's own pass (metre ok and judge score >= threshold);
  - metre_pass: the best attempt has no metre failures (verse only);
  - judge: the judge's score for the best attempt;
  - style_match: the style classifier (evals/style_eval.py) attributes the piece to the requested author.
    Skipped for 'plain', which has no author.

Needs ANTHROPIC_API_KEY. Costs model calls: up to 2 per iteration per run.

Run:  python -m evals.gen_eval [--iters 3] [--limit 4]
"""
import argparse
import pickle
import tempfile
from pathlib import Path

from latinwriter import agent
from latinwriter.config import ROOT

from .common import save

MODEL_PATH = ROOT / "data" / "style_model.pkl"

CASES = [
    ("hexameter", "vergil", "a storm that scatters a fleet at sea"),
    ("hexameter", "vergil", "a city founded by an exiled hero"),
    ("hexameter", "vergil", "a god who sends a messenger to a queen"),
    ("hexameter", "ovid", "a nymph who turns into a tree"),
    ("hexameter", "ovid", "a king who is turned into an animal"),
    ("elegiac", "ovid", "a lover who waits at a closed door"),
    ("elegiac", "ovid", "a poet complaining about an unfaithful mistress"),
    ("elegiac", "ovid", "a farewell to a friend before a long journey"),
    ("prose", "caesar", "the army crossing a river under fire"),
    ("prose", "caesar", "a report on the tribes of Gaul"),
    ("prose", "cicero", "an accusation against a corrupt magistrate"),
    ("prose", "cicero", "a defence of a friend before the senate"),
]
AUTHOR = {"vergil": "Vergil", "ovid": "Ovid", "caesar": "Caesar", "cicero": "Cicero"}


def _style_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("run `python -m evals.style_eval` first to build the style classifier")
    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def run(iters: int = 3, limit: int | None = None) -> dict:
    model = _style_model()
    cases = CASES[:limit] if limit else CASES
    rows = []
    workdir = ROOT / "data" / "gen_eval_runs"
    for form, style, theme in cases:
        for exemplars in (True, False):
            tag = f"{form}_{style}_{'ex' if exemplars else 'noex'}_{len(rows)}"
            report = agent.write(
                form=form, style=style, theme=theme, out=workdir / tag,
                max_iters=iters, exemplars=exemplars, log=lambda *_: None,
            )
            best_idx = report["best_iteration"]
            best = report["attempts"][best_idx] if best_idx is not None else None
            piece = "\n".join(best["latin"]) if best else ""
            if best and style in AUTHOR and piece:
                predicted = model.predict([piece.lower()])[0]
                style_match = predicted == AUTHOR[style]
            else:
                predicted, style_match = None, None
            row = {
                "form": form, "style": style, "theme": theme, "exemplars": exemplars,
                "result": report["result"],
                "iterations": len(report["attempts"]),
                "metre_pass": (best["meter_failures"] == 0) if (best and form != "prose") else None,
                "judge": best["judge"].get("score") if best else None,
                "predicted_author": predicted,
                "style_match": style_match,
                "piece": piece,
            }
            rows.append(row)
            print(f"{form:<10} {style:<7} ex={exemplars!s:<5} result={row['result']:<16} "
                  f"metre={row['metre_pass']} judge={row['judge']} style={row['predicted_author']}")

    def summary(subset):
        if not subset:
            return {}
        verse = [r for r in subset if r["metre_pass"] is not None]
        styled = [r for r in subset if r["style_match"] is not None]
        judged = [r["judge"] for r in subset if r["judge"] is not None]
        return {
            "runs": len(subset),
            "pass_rate": round(sum(r["result"] == "ok" for r in subset) / len(subset), 3),
            "metre_pass_rate": round(sum(bool(r["metre_pass"]) for r in verse) / len(verse), 3) if verse else None,
            "style_match_rate": round(sum(bool(r["style_match"]) for r in styled) / len(styled), 3) if styled else None,
            "mean_judge": round(sum(judged) / len(judged), 1) if judged else None,
            "mean_iterations": round(sum(r["iterations"] for r in subset) / len(subset), 2),
        }

    payload = {
        "eval": "generation",
        "iters": iters,
        "summary": {
            "exemplars_on": summary([r for r in rows if r["exemplars"]]),
            "exemplars_off": summary([r for r in rows if not r["exemplars"]]),
        },
        "rows": rows,
    }
    save("gen_eval.json", payload)
    print(payload["summary"])
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="only the first N cases")
    args = ap.parse_args()
    if not __import__("os").environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set; the generation eval needs the model.")
    run(iters=args.iters, limit=args.limit)
