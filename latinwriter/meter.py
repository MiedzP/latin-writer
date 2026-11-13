"""Heuristic metrical check for Latin verse: dactylic hexameter and elegiac couplets.

This is NOT a full scansion. It implements:
  - elision (final vowel or -m before a vowel or h is dropped)
  - consonantal i/u at word start (iam, uenit)
  - diphthongs (ae oe au ei eu oi) are long
  - the position rule (a vowel followed by two consonants is long), with x and z counting as two
  - muta cum liquida (e.g. -tr-, -pl-) is ambiguous, so it is treated as unknown

Every other vowel is 'U' (unknown: long or short). The result is a sanity check that catches
wrong syllable counts, heavy syllables in short positions, and missing feet. It does not
know real vowel quantities, so a pass means "plausible", not "scans correctly".
"""
import itertools
import re
from dataclasses import dataclass, field

VOWELS = "aeiouy"
DIPHTHONGS = ("ae", "oe", "au", "ei", "eu", "oi")
MUTA = set("bcdgptf")
LIQUID = set("lr")

# Slot patterns: L = long slot (anything allowed), S = short slot (only U allowed),
# X = anceps (anything). Heavy syllables can therefore only sit in long slots.
FOOT = {"D": "LSS", "S": "LL"}
HEXAMETER_PATTERNS = [
    "".join(FOOT[f] for f in combo) + "LSS" + "LX"
    for combo in itertools.product("DS", repeat=4)
]
PENTAMETER_HALF = "LSSLSSL"
PENTAMETER_PATTERN = PENTAMETER_HALF + PENTAMETER_HALF

SHOW = {"L": "-", "S": "u", "X": "x"}


@dataclass
class Scansion:
    line: str
    form: str
    ok: bool
    syllables: int
    weights: str = ""           # H = heavy (certain), U = unknown
    pattern: str = ""           # matched pattern, shown as - u x
    reason: str = ""
    syllable_text: list[str] = field(default_factory=list)


def _normalise(line: str) -> list[str]:
    text = line.lower().replace("y", "i")          # j and v stay as consonants
    text = re.sub(r"[^a-z\s]", " ", text)
    return text.split()


def _elide(words: list[str]) -> list[str]:
    words = list(words)
    for i in range(len(words) - 1):
        w, nxt = words[i], words[i + 1]
        if not nxt or not w:
            continue
        next_starts_vowel = nxt[0] in VOWELS
        next_starts_h = nxt[0] == "h"
        if not (next_starts_vowel or next_starts_h):
            continue
        if w.endswith("m"):
            words[i] = w[:-1]
        elif w[-1] in VOWELS:
            words[i] = re.sub(r"[aeiouy]+$", "", w)
        else:
            continue
        if next_starts_h:
            words[i + 1] = nxt[1:]
    return [w for w in words if w]


def _mark_consonants(word: str) -> str:
    word = re.sub(r"^i(?=[aeiouy])", "j", word)     # consonantal i (iam) reads as j
    return word.replace("qu", "Q").replace("ch", "K").replace("ph", "F").replace("th", "T")


_DISPLAY = {"Q": "qu", "K": "ch", "F": "ph", "T": "th"}


def _nuclei(s: str):
    """Yield (start, end, is_diphthong) for each vowel nucleus in s."""
    for m in re.finditer(r"[aeiouy]+", s):
        grp, base, i = m.group(), m.start(), 0
        while i < len(grp):
            if grp[i:i + 2] in DIPHTHONGS and i + 2 <= len(grp):
                yield base + i, base + i + 2, True
                i += 2
            else:
                yield base + i, base + i + 1, False
                i += 1


def weigh(line: str) -> tuple[list[str], list[str]]:
    """Return (weights, syllable_texts) for one line."""
    words = _elide(_normalise(line))
    s = "".join(_mark_consonants(w) for w in words)
    nuclei = list(_nuclei(s))
    weights, texts = [], []
    for k, (start, end, diph) in enumerate(nuclei):
        nxt_start = nuclei[k + 1][0] if k + 1 < len(nuclei) else len(s)
        coda = s[end:nxt_start]
        count = sum(2 if c in "xz" else 1 for c in coda)
        if diph:
            w = "H"
        elif count >= 3:
            w = "H"
        elif count == 2:
            ambiguous = len(coda) == 2 and coda[0] in MUTA and coda[1] in LIQUID
            w = "U" if ambiguous else "H"
        else:
            w = "U"
        weights.append(w)
        texts.append(''.join(_DISPLAY.get(c, c) for c in s[start:nxt_start]))
    return weights, texts


def _fits(weights: str, pattern: str) -> bool:
    if len(weights) != len(pattern):
        return False
    for w, p in zip(weights, pattern):
        if p == "S" and w == "H":
            return False
    return True


def scan(line: str, form: str = "hexameter") -> Scansion:
    weights, texts = weigh(line)
    n = len(weights)
    res = Scansion(line=line.strip(), form=form, ok=False, syllables=n,
                   weights="".join(weights), syllable_text=texts)

    if form == "prose":
        res.ok = True
        res.reason = "prose: no metre check"
        return res

    if form == "hexameter":
        candidates = HEXAMETER_PATTERNS
        if not 12 <= n <= 17:
            res.reason = f"{n} syllables; a hexameter has 12-17"
            return res
    elif form == "pentameter":
        candidates = [PENTAMETER_PATTERN]
        if n != len(PENTAMETER_PATTERN):
            res.reason = f"{n} syllables; a pentameter has {len(PENTAMETER_PATTERN)}"
            return res
    else:
        raise ValueError(f"unknown form: {form}")

    for pat in candidates:
        if len(pat) == n and _fits("".join(weights), pat):
            res.ok = True
            res.pattern = "".join(SHOW[c] for c in pat)
            res.reason = "fits a metrical pattern"
            return res

    heavy_in_short = [i for i, w in enumerate(weights) if w == "H"]
    res.reason = (
        f"{n} syllables, no hexameter/pentameter foot pattern fits; "
        f"heavy syllables at {heavy_in_short or 'none'}"
    )
    return res


def scan_couplet(lines: list[str]) -> list[Scansion]:
    """Elegiac couplets alternate hexameter and pentameter."""
    return [scan(line, "hexameter" if i % 2 == 0 else "pentameter") for i, line in enumerate(lines)]
