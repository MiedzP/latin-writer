FORMS = {
    "hexameter": (
        "dactylic hexameter, epic verse. Every line has six feet: five dactyls or spondees, then a "
        "dactyl and a spondee. Use a caesura in the third or fourth foot. Write verses, not prose."
    ),
    "elegiac": (
        "elegiac couplets. Lines alternate: a dactylic hexameter, then a pentameter (two halves, each "
        "two dactyls and a long syllable). Write whole couplets, so the number of lines is even."
    ),
    "prose": "continuous prose, several sentences, with the sentence rhythm of a formal Latin author.",
}

STYLES = {
    "vergil": "grave, dense, allusive, with epic similes and archaic touches.",
    "ovid": "witty, polished, with antithesis, epigram and a conversational tone.",
    "cicero": "periodic sentences, clauses held in suspense, ablative absolutes, ut and cum clauses.",
    "caesar": "plain, short declarative sentences, indirect statement, little ornament.",
    "plain": "clear classroom Latin with simple vocabulary and standard grammar.",
}

GENERATOR_SYSTEM = """You write original Latin for a learner-facing tool. You know classical Latin grammar and metre well.

Write {form}: {form_rule}
Style: {style_rule}
Use Classical Latin spelling (no j, v as u, no k). Avoid rare words where a common one works.

Output format, exactly:
LATIN:
<the Latin, one verse or sentence per line>
TRANSLATION:
<a faithful English translation, one line per Latin line>

Do not add notes, titles or commentary outside that format."""

GENERATOR_FEEDBACK = """Last attempt needs fixing.

Metre problems (checked by code, approximate, quantity is heuristic):
{meter}

Grammar and style problems (from a Latin judge, score {score}/100):
{judge}

Rewrite the whole piece in the same format. Keep lines that were fine, and change only what is needed."""

JUDGE_SYSTEM = """You are an expert in classical Latin grammar and style. You check a piece of Latin written by a machine for learners.

Check: case, agreement, verb tense and mood, word order that a Latin reader would find wrong, non-classical vocabulary or forms,
and whether the requested style is followed. Ignore metre; a separate program checks it.
Do not penalise poetic liberty that real Latin poets used.

Reply ONLY with JSON:
{
  "score": 0-100 integer for grammatical correctness and style fit,
  "errors": [{"line": 1, "problem": "...", "fix": "a corrected Latin phrase"}],
  "strengths": ["..."]
}
If the Latin is correct, return an empty errors list and a high score."""
