"""Collect and parse a small Latin corpus from thelatinlibrary.com.

Politeness: one request per second, a descriptive User-Agent, and every page cached in
data/raw/ so each page is fetched once. The site's robots.txt is empty (no disallow rules).
The texts are classical works; the corpus stays local (data/ is git-ignored).

Output: data/corpus.jsonl, one record per verse line or prose sentence:
  {"author", "work", "section", "kind": "verse"|"prose", "split": "train"|"test", "text"}
Split rule: sections where section % 4 == 0 are the held-out test set, so no section is
split between train and test.
"""
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import ROOT

BASE = "https://www.thelatinlibrary.com/"
USER_AGENT = "latin-writer-research/0.1 (personal research project; contact via GitHub MiedzP)"
DATA = ROOT / "data"
RAW = DATA / "raw"

SOURCES = [
    # (author, work, kind, [(section, path), ...])
    ("Vergil", "Aeneid", "verse", [(i, f"vergil/aen{i}.shtml") for i in range(1, 13)]),
    ("Ovid", "Metamorphoses", "verse", [(i, f"ovid/ovid.met{i}.shtml") for i in range(1, 16)]),
    ("Caesar", "Gallic War", "prose", [(i, f"caesar/gallic/gall{i}.shtml") for i in range(1, 9)]),
    ("Cicero", "Speeches", "prose", [
        (1, "cicero/cat.shtml"), (2, "cicero/milo.shtml"), (3, "cicero/murena.shtml"), (4, "cicero/flacco.shtml"),
    ]),
]


def fetch(path: str, delay: float = 1.0) -> str:
    """Return the HTML for `path`, from cache if present, otherwise from the site (rate-limited)."""
    RAW.mkdir(parents=True, exist_ok=True)
    cache = RAW / path.replace("/", "__")
    if cache.exists():
        return cache.read_text(encoding="utf-8", errors="replace")
    req = urllib.request.Request(BASE + path, headers={"User-Agent": USER_AGENT})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            cache.write_text(body, encoding="utf-8")
            time.sleep(delay)
            return body
        except urllib.error.URLError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"could not fetch {path}")


def _clean(fragment: str) -> str:
    fragment = re.sub(r"<span[^>]*>\s*\d+\s*</span>", "", fragment)   # line-number markers
    fragment = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.I)
    fragment = re.sub(r"<[^>]+>", " ", fragment)
    text = html.unescape(fragment).replace("\xa0", " ").replace("—", " ").replace("–", " ")
    text = text.replace("<", "").replace(">", "")          # editorial supplements, e.g. &lt;sicut&gt;
    return text


def parse_page(body: str, kind: str) -> list[str]:
    """Return the verse lines (kind == 'verse') or prose sentences (kind == 'prose') of a page."""
    blocks = re.findall(r"<p[^>]*>(.*?)</p>", body, flags=re.S | re.I)
    out: list[str] = []
    for block in blocks:
        text = _clean(block)
        lines = []
        for ln in text.split("\n"):
            ln = re.sub(r"\[\s*\d+\s*\]", " ", ln).strip()   # Caesar chapter markers: [ 21 ]
            ln = re.sub(r"\s+\d+$", "", ln).strip()          # trailing verse numbers: 410
            lines.append(ln)
        lines = [ln for ln in lines if ln and re.search(r"[A-Za-zāēīōūĀĒĪŌŪ]", ln)]
        if not lines:
            continue
        if kind == "verse":
            # verse blocks have one line per <br>; keep only blocks that look like verse
            if len(lines) >= 2 and all(len(ln.split()) <= 14 for ln in lines):
                out.extend(lines)
        else:
            prose = " ".join(lines)
            sentences = re.split(r"(?<=[.!?;])\s+(?=[A-ZĀĒĪŌŪ\"'(])", prose)
            out.extend(s.strip() for s in sentences if len(s.split()) >= 4)
    return out


def build(max_pages: int | None = None) -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    out_path = DATA / "corpus.jsonl"
    records = 0
    pages = 0
    with out_path.open("w", encoding="utf-8") as f:
        for author, work, kind, pages_list in SOURCES:
            for section, path in pages_list:
                if max_pages is not None and pages >= max_pages:
                    break
                try:
                    body = fetch(path)
                except RuntimeError as err:
                    print(f"skip {path}: {err}")
                    continue
                pages += 1
                split = "test" if section % 4 == 0 else "train"
                for text in parse_page(body, kind):
                    rec = {"author": author, "work": work, "section": section,
                           "kind": kind, "split": split, "text": text}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    records += 1
    print(f"pages {pages}, records {records} -> {out_path}")
    return out_path


def load(split: str | None = None, kind: str | None = None) -> list[dict]:
    path = DATA / "corpus.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if split:
        rows = [r for r in rows if r["split"] == split]
    if kind:
        rows = [r for r in rows if r["kind"] == kind]
    return rows
