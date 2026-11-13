"""Two-pass Latin judge.

Pass 1 judges the text on its own. If the judge reports low confidence, pass 2 retrieves the most
relevant pages of the local grammar book and judges again with those pages as reference. The book
is only consulted when the first answer is unsure; it is not read in full for every item.

If the grammar index is not available (no local copy of the book), the judge runs single-pass.
"""
import json
from pathlib import Path

from .config import JUDGE_MODEL, ROOT
from .llm import ask, extract_json

JUDGE_SYSTEM = """You are an expert in classical Latin grammar and style. You check a piece of Latin written by a machine for learners.

Check: case, agreement, verb tense and mood, word order a Latin reader would find wrong, non-classical vocabulary or forms,
and whether the requested style is followed. Ignore metre; a separate program checks it.
Do not penalise poetic liberty that real Latin poets used.

Also say how sure you are. Use "low" confidence when you are unsure whether a construction is correct,
when it is a rare or disputed construction, or when you would need to check a grammar rule.

Reply ONLY with JSON:
{
  "score": 0-100 integer for grammatical correctness and style fit,
  "errors": [{"line": 1, "problem": "...", "fix": "a corrected Latin phrase"}],
  "strengths": ["..."],
  "confidence": "high" or "low",
  "reference_query": "a short phrase naming the grammatical construction at issue, e.g. 'purpose clause with ut and subjunctive'"
}
If the Latin is correct, return an empty errors list and a high score."""

REFERENCE_NOTE = """

Reference pages from a Latin grammar (use them as the authority on the rule in question; if they do not cover it, say so and keep your confidence low):
{pages}"""


def _grammar_index():
    """Return the grammar index, or None if the local book has not been extracted."""
    if not (ROOT / "data" / "grammar" / "sections.jsonl").exists():
        return None
    try:
        from .grammar import GrammarIndex
        return GrammarIndex()
    except Exception:
        return None


def _ask_judge(latin: str, form: str, style: str, theme: str, reference: list[dict] | None,
               model: str) -> dict:
    user = f"Form: {form}. Style: {style}. Requested theme: {theme}.\n\nLatin text:\n{latin}"
    system = JUDGE_SYSTEM
    if reference:
        pages = "\n\n".join(f"[page {r['page']}, {r['part']}]\n{r['text'][:3000]}" for r in reference)
        system += REFERENCE_NOTE.format(pages=pages)
    last_error = None
    for _ in range(2):  # one retry: models sometimes return malformed JSON
        reply = ask([{"role": "user", "content": user}], system, model=model, max_tokens=2000)
        try:
            return extract_json(reply)
        except (ValueError, json.JSONDecodeError) as err:
            last_error = err
    return {"score": 0, "errors": [{"line": 0, "problem": f"judge reply was not valid JSON ({last_error})",
                                    "fix": ""}],
            "strengths": [], "confidence": "low", "reference_query": ""}


def judge_latin(latin: str, form: str, style: str, theme: str, model: str = JUDGE_MODEL,
                index=None) -> dict:
    """Judge a Latin text. Escalates to the grammar book only when pass 1 reports low confidence."""
    first = _ask_judge(latin, form, style, theme, None, model)
    if first.get("confidence", "high") != "low":
        return dict(first, escalated=False)
    if index is None:
        index = _grammar_index()
    if index is None:
        return dict(first, escalated=False, note="low confidence, but no grammar book is available")
    query = f"{first.get('reference_query', '')} {latin}"
    refs = index.retrieve(query, k=3)
    if not refs:
        return dict(first, escalated=False, note="low confidence, and no relevant pages were found")
    second = _ask_judge(latin, form, style, theme, refs, model)
    return dict(second, escalated=True, first_pass_score=first.get("score"),
                reference_pages=[r["page"] for r in refs])
