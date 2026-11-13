"""Extract grammar sections and example sentences from the Latin grammar PDF's text.

Input:  data/grammar/latin_grammar.txt (made from the PDF with pypdf; local only, not in git)
Output: data/grammar/examples.jsonl  - Latin example sentences with translation, citation, page
        data/grammar/sections.jsonl  - one chunk of text per page, tagged with its running head
                                       and its major part (verbal system, cases, clauses, ...)

The book prints each usage example as:  • <Latin sentence>  <ENGLISH TRANSLATION>  (Author, Work).
Examples are gold for grammaticality: they are authentic Latin that the book uses to illustrate a rule.
"""
import json
import re
from pathlib import Path

from .config import ROOT

GRAMMAR_DIR = ROOT / "data" / "grammar"
TEXT = GRAMMAR_DIR / "latin_grammar.txt"

# Major parts, by the first printed page of each part (from the book's contents page).
PARTS = [
    (1, "nouns and adjectives"),
    (55, "pronouns"),
    (100, "adverbs"),
    (130, "verbal system"),
    (221, "syntax of cases"),
    (278, "syntax of clauses"),
]

PAGE_RE = re.compile(r"^=== PAGE (\d+) ===$", re.M)
# Latin is lowercase-ish; the translation is in capitals. Allow a short prefix like "2/ " or "[280]".
EXAMPLE_RE = re.compile(
    r"•\s*(?P<latin>[^•\n]+?)\s{2,}(?P<english>[A-Z][A-Z0-9 ,.'’;:!?\-—()\"“”&]+?)\s*(?:\((?P<cite>[^)]{3,80})\))?\s*(?=\n|•|$)"
)


def _pages(text: str) -> list[tuple[int, str]]:
    parts = PAGE_RE.split(text)
    # split gives ['', '1', 'text', '2', 'text', ...]
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]


def part_of(page: int) -> str:
    name = PARTS[0][1]
    for start, label in PARTS:
        if page >= start:
            name = label
    return name


def build() -> dict:
    text = TEXT.read_text(encoding="utf-8")
    pages = _pages(text)
    sections, examples = [], []
    for page, body in pages:
        clean = re.sub(r"[ \t]+", " ", body).strip()
        sections.append({"page": page, "part": part_of(page), "text": clean})
        for m in EXAMPLE_RE.finditer(body):
            latin = re.sub(r"\s+", " ", m.group("latin")).strip(" .")
            english = re.sub(r"\s+", " ", m.group("english")).strip()
            # keep plausible Latin sentences: several words, mostly Latin letters, not a heading
            words = latin.split()
            if len(words) < 2 or len(latin) > 220 or re.search(r"[A-Z]{4,}", latin):
                continue
            examples.append({"page": page, "part": part_of(page), "latin": latin,
                             "english": english, "cite": m.group("cite") or ""})

    GRAMMAR_DIR.mkdir(parents=True, exist_ok=True)
    with (GRAMMAR_DIR / "sections.jsonl").open("w", encoding="utf-8") as f:
        for s in sections:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    with (GRAMMAR_DIR / "examples.jsonl").open("w", encoding="utf-8") as f:
        for e in examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    summary = {"pages": len(sections), "examples": len(examples)}
    print(summary)
    return summary


def load_examples(part: str | None = None) -> list[dict]:
    path = GRAMMAR_DIR / "examples.jsonl"
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if part is None or r["part"] == part]


def load_sections() -> list[dict]:
    path = GRAMMAR_DIR / "sections.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


class GrammarIndex:
    """TF-IDF index over the book's pages. retrieve() returns the pages most relevant to a query.

    This is how the book is 'trained on' here: no weights change. The judge can be given the
    retrieved pages as reference, and the grammar eval can compare judging with and without them.
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.sections = [s for s in load_sections() if len(s["text"]) > 200]
        self.vectorizer = TfidfVectorizer(lowercase=True, sublinear_tf=True, stop_words="english",
                                          token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b")
        self.matrix = self.vectorizer.fit_transform([s["text"] for s in self.sections])

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        q = self.vectorizer.transform([query])
        scores = (self.matrix @ q.T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [dict(self.sections[i], score=round(float(scores[i]), 3)) for i in order if scores[i] > 0]


if __name__ == "__main__":
    build()
