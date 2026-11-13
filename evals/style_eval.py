"""Author-attribution eval: can a classifier tell the authors apart on held-out sections?

Two tasks:
  - all four authors (genre and author together);
  - verse only (Vergil vs Ovid) and prose only (Caesar vs Cicero), which test style rather than genre.

The classifier (character n-grams, logistic regression) is saved to data/style_model.pkl. The
generation eval uses it to check whether a generated piece reads as the requested author.

Run:  python -m evals.style_eval
"""
import pickle
import random
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from latinwriter.config import ROOT

from .common import clean, held_out, save, train

MODEL_PATH = ROOT / "data" / "style_model.pkl"
PER_CLASS_TRAIN = 2000   # cap so one large author does not dominate


def _balanced(rows: list[dict], per_class: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by = {}
    for r in rows:
        by.setdefault(r["author"], []).append(r)
    out = []
    for author, items in by.items():
        rng.shuffle(items)
        out += items[:per_class]
    return out


def _fit(rows: list[dict], label_authors: list[str]):
    pipe = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True, min_df=2),
        LogisticRegression(max_iter=2000, C=5.0),
    )
    x = [r["text"].lower() for r in rows]
    y = [r["author"] for r in rows]
    pipe.fit(x, y)
    return pipe


def _evaluate(name: str, authors: list[str], kinds: list[str]) -> dict:
    tr = [r for r in train(None) if r["author"] in authors and r["kind"] in kinds] if False else [
        r for kind in kinds for r in train(kind) if r["author"] in authors]
    te = [r for kind in kinds for r in held_out(kind) if r["author"] in authors]
    tr = _balanced(tr, PER_CLASS_TRAIN)
    model = _fit(tr, authors)
    pred = model.predict([r["text"].lower() for r in te])
    gold = [r["author"] for r in te]
    labels = sorted(authors)
    return {
        "task": name,
        "train_size": len(tr),
        "test_size": len(te),
        "train_counts": dict(Counter(r["author"] for r in tr)),
        "test_counts": dict(Counter(gold)),
        "accuracy": round(accuracy_score(gold, pred), 4),
        "macro_f1": round(f1_score(gold, pred, average="macro"), 4),
        "labels": labels,
        "confusion_matrix": confusion_matrix(gold, pred, labels=labels).tolist(),
    }


def run() -> dict:
    tasks = [
        ("all_four_authors", ["Vergil", "Ovid", "Caesar", "Cicero"], ["verse", "prose"]),
        ("verse_vergil_vs_ovid", ["Vergil", "Ovid"], ["verse"]),
        ("prose_caesar_vs_cicero", ["Caesar", "Cicero"], ["prose"]),
    ]
    results = [_evaluate(*t) for t in tasks]

    # Save the four-author model (trained on all training data) for the generation eval.
    tr = _balanced([r for kind in ["verse", "prose"] for r in train(kind)], PER_CLASS_TRAIN)
    model = _fit(tr, ["Vergil", "Ovid", "Caesar", "Cicero"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MODEL_PATH.open("wb") as f:
        pickle.dump(model, f)

    report = {"eval": "style_attribution", "results": results, "model": str(MODEL_PATH)}
    save("style_eval.json", report)
    for r in results:
        print(f"{r['task']:<26} accuracy {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
              f"(train {r['train_size']}, test {r['test_size']})")
    return report


if __name__ == "__main__":
    run()
