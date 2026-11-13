"""Finite-verb detector for the verb eval layer.

Detection is by lexicon, not by grammar: a token counts as a verb if it is a known finite verb form.
The lexicon covers the copula (esse) completely and a set of very common verbs. It is partial, so
the verb eval reports how often a sentence with a real verb is actually caught (coverage).

Distinguishing the three cases the eval cares about:
  - explicit: a finite verb form is present in the text;
  - implied:  no verb is written, but the sentence is grammatical (ellipsis, e.g. omitted "est");
  - missing:  no verb is written and the sentence is not grammatical without one.
The detector only supplies the "explicit" call. Implied versus missing needs a human judgment,
which is what the judging step in evals/verb_eval.py collects.
"""
import re
import unicodedata

ESSE = """
sum es est sumus estis sunt eram eras erat eramus eratis erant ero eris erit erimus eritis erunt
sim sis sit simus sitis sint essem esses esset essemus essetis essent fui fuisti fuit fuimus fuistis
fuerunt fuere fueram fuerat fuerant fuerit fuerint fuisset fuissent
"""

COMMON = """
dicit dicunt dixit dixerunt dicet dicebat dicebant dicitur
facit faciunt fecit fecerunt faciet faciebat fecerat
habet habent habuit habuerunt habebat habebant habebit
videt vident vidit viderunt videbat videbant
it eunt ibat ibant ivit ierunt ibit
venit veniunt venerunt veniebat venerat
dat dant dedit dederunt dabat dabant dabit
mittit mittunt misit miserunt mittebat
capit capiunt cepit ceperunt capiebat
fert ferunt tulit tulerunt ferebat
ducit ducunt duxit duxerunt ducebat
ponit ponunt posuit posuerunt ponebat
respondit respondent iussit iusserunt coepit coeperunt incipit incipiunt incepit
potest possunt potuit potuerunt poterat
vult volunt voluit voluerunt noluit nolunt
audit audiunt audivit audiverunt audiebat
tenet tenent tenuit tenuerunt
ostendit ostenderunt cognovit cognoverunt
ait aiunt inquit inquiunt
appellat appellant pugnat pugnant pugnaverunt
committit committunt putat putant putavit
vocat vocant vocavit nuntiat nuntiant
reliquit reliquerunt relinquit cedit cesserunt
movit movet petit petunt petiit reperit invenit inveniunt invenerunt
fugit fugerunt interficit interfecit instituit constituit
laudat laudant amat amant amavit amaverunt
timet timent timuit timuerunt timebat
sedet sedent sedebat manet manent mansit manserunt
"""

LEXICON = {w for block in (ESSE, COMMON) for w in block.split()}


def _ascii(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", word.lower()) if unicodedata.category(c) != "Mn")


def find_verbs(text: str) -> list[str]:
    """Return the lexicon verb forms found in the text (in order)."""
    tokens = re.findall(r"[A-Za-zāēīōūĀĒĪŌŪ]+", text)
    return [t for t in tokens if _ascii(t) in LEXICON]


def has_explicit_verb(text: str) -> bool:
    return bool(find_verbs(text))


def remove_verb(text: str, index: int = 0) -> tuple[str, str] | None:
    """Delete the index-th lexicon verb from the text. Returns (new_text, removed_word) or None."""
    verbs = find_verbs(text)
    if index >= len(verbs):
        return None
    target = verbs[index]
    # Remove the matching token occurrence (the index-th occurrence of that word in the text).
    occurrences = [m for m in re.finditer(r"[A-Za-zāēīōūĀĒĪŌŪ]+", text) if _ascii(m.group()) in LEXICON]
    m = occurrences[index]
    new_text = (text[: m.start()] + text[m.end():]).replace("  ", " ").replace(" ,", ",").replace(" .", ".").strip()
    return new_text, target
