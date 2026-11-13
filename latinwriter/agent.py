"""Write Latin verse or prose: generate, check metre in code, judge grammar with a model, revise.

Each iteration:
  1. the generator writes the piece (or revises it from feedback);
  2. for verse, every line is checked by the metre heuristic (code, no model);
  3. the judge reviews the Latin for grammar, idiom and style (model);
  4. stop when the metre passes and the judge score reaches the threshold, or at the cap.
The best attempt is kept. Metre failures are penalised in the score, so a grammatical but
unmetrical attempt does not beat a metrical one.
"""
import json
import random
from pathlib import Path
from typing import Callable

from .llm import ask, extract_json, parse_output
from .meter import scan, scan_couplet
from .prompts import FORMS, GENERATOR_FEEDBACK, GENERATOR_SYSTEM, JUDGE_SYSTEM, STYLES
from .config import JUDGE_MODEL

Log = Callable[[str], None]
METER_PENALTY = 10

# Writer styles that correspond to a corpus author (exemplars come from that author's training split).
STYLE_AUTHOR = {"vergil": "Vergil", "ovid": "Ovid", "cicero": "Cicero", "caesar": "Caesar"}


def exemplar_block(author: str, form: str, n: int = 8, seed: int = 0) -> str:
    """Sample lines of the author's TRAINING split, for the generator's system prompt.

    The held-out test sections are never used here, so the generation eval stays clean.
    """
    from .corpus import load
    kind = "prose" if form == "prose" else "verse"
    try:
        pool = [r["text"] for r in load(split="train", kind=kind) if r["author"] == author]
    except FileNotFoundError:
        return ""
    if not pool:
        return ""
    picks = random.Random(seed).sample(pool, min(n, len(pool)))
    header = f"\n\nExamples of {author}'s own Latin, for style and vocabulary only; do not copy them:\n"
    return header + "\n".join(f"- {p}" for p in picks)


def _metre_results(lines: list[str], form: str):
    if form == "hexameter":
        return [scan(ln, "hexameter") for ln in lines]
    if form == "elegiac":
        return scan_couplet(lines)
    return [scan(ln, "prose") for ln in lines]


def _judge(latin: str, form: str, style: str, theme: str) -> dict:
    user = (
        f"Form: {form}. Style: {style}. Requested theme: {theme}.\n\n"
        f"Latin text:\n{latin}"
    )
    reply = ask([{"role": "user", "content": user}], JUDGE_SYSTEM, model=JUDGE_MODEL, max_tokens=2000)
    return extract_json(reply)


def _meter_feedback(results) -> str:
    bad = [r for r in results if not r.ok]
    if not bad:
        return "none"
    return "\n".join(
        f"- line {i + 1}: \"{r.line}\" — {r.reason}" for i, r in enumerate(results) if not r.ok
    )


def write(
    form: str,
    style: str,
    theme: str,
    out: Path,
    lines: int = 6,
    max_iters: int = 4,
    threshold: int = 80,
    exemplars: bool = True,
    log: Log = print,
) -> dict:
    if form not in FORMS:
        raise ValueError(f"form must be one of {sorted(FORMS)}")
    if style not in STYLES:
        raise ValueError(f"style must be one of {sorted(STYLES)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    system = GENERATOR_SYSTEM.format(form=form, form_rule=FORMS[form], style_rule=STYLES[style])
    if exemplars and style in STYLE_AUTHOR:
        system += exemplar_block(STYLE_AUTHOR[style], form)
    request = f"Theme: {theme}. Length: about {lines} {'lines' if form != 'prose' else 'sentences'}."
    messages = [{"role": "user", "content": request}]

    attempts: list[dict] = []
    best: dict | None = None

    for i in range(max_iters):
        reply = ask(messages, system)
        messages.append({"role": "assistant", "content": reply})
        latin_lines, translation = parse_output(reply)
        latin = "\n".join(latin_lines)

        results = _metre_results(latin_lines, form) if form != "prose" else []
        meter_fail = [r for r in results if not r.ok]

        judge = _judge(latin, form, style, theme)
        judge_score = int(judge.get("score", 0))
        score = max(0, judge_score - METER_PENALTY * len(meter_fail))

        att = {
            "iteration": i,
            "latin": latin_lines,
            "translation": translation,
            "meter": [r.__dict__ | {"syllable_text": r.syllable_text} for r in results],
            "meter_failures": len(meter_fail),
            "judge": judge,
            "score": score,
        }
        attempts.append(att)
        log(f"[{i}] judge {judge_score}/100, metre failures {len(meter_fail)}, score {score}")

        if best is None or score > best["score"]:
            best = att

        if not meter_fail and judge_score >= threshold:
            break

        errors = "\n".join(
            f"- line {e.get('line', '?')}: {e.get('problem', '')} (fix: {e.get('fix', '')})"
            for e in judge.get("errors", [])
        ) or "none"
        messages.append({
            "role": "user",
            "content": GENERATOR_FEEDBACK.format(
                meter=_meter_feedback(results) if results else "not applicable (prose)",
                score=judge_score,
                judge=errors,
            ),
        })

    passed = bool(best) and best["meter_failures"] == 0 and int(best["judge"].get("score", 0)) >= threshold
    report = {
        "form": form, "style": style, "theme": theme, "exemplars": exemplars,
        "result": "ok" if passed else "below_threshold",
        "best_iteration": best["iteration"] if best else None,
        "attempts": attempts,
    }
    if best:
        (out / "piece.txt").write_text(
            "\n".join(best["latin"]) + "\n\n" + best["translation"] + "\n", encoding="utf-8"
        )
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"done: best = {report['best_iteration']}, result = {report['result']}")
    return report
