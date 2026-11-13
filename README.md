# latin-writer

An agent that writes original Latin verse (dactylic hexameter, elegiac couplets) and prose,
and checks what it writes.

Each attempt goes through three steps:

1. **Generate** — the model writes the piece in a requested form and style.
2. **Metre check (code)** — every verse line is checked by a heuristic scanner: syllable count,
   elision, diphthongs, the position rule, and whether heavy syllables fall in long positions of a
   hexameter or pentameter pattern.
3. **Grammar judge (model)** — a second model call reviews case, agreement, tense, idiom and style.

Failures from both checks go back to the generator, and the loop runs until the piece passes or the
cap is reached. The best attempt is kept, and metre failures lower its score.

## Setup

```
python -m pip install -r requirements.txt
set ANTHROPIC_API_KEY=...        # only needed for `write`; `check` runs offline
```

## Usage

```
python -m latinwriter check mylines.txt --form hexameter     # metre-check your own lines, no key needed
python -m latinwriter write --form hexameter --style vergil --theme "a storm at sea" --out runs\storm
python -m latinwriter write --form elegiac --style ovid --theme "a lover's quarrel" --out runs\quarrel
python -m latinwriter write --form prose --style caesar --theme "the march to the Rhine" --out runs\rhine
```

Styles: `vergil`, `ovid`, `cicero`, `caesar`, `plain`. Forms: `hexameter`, `elegiac`, `prose`.

Output in `--out`: `piece.txt` (Latin plus translation) and `report.json` (every attempt, scansion,
judge notes, and the score).

## What is and isn't checked

- **Metre is approximate.** The scanner doesn't know real vowel quantities. Vowels it can't classify
  are marked unknown and allowed in either slot, so the check rejects lines with wrong length or
  misplaced heavy syllables. It will also pass some lines a metrician would reject. Its tests
  (`tests/test_meter.py`) cover three known Vergil hexameters, a short fragment, and an English line.
- **Grammar and style come from a model judge.** It can be wrong, and it can miss errors. Output
  should be checked by someone who reads Latin before it is shown to students.
- **Poetry quality is not measured.** The loop gets metrically correct, grammatical verse. Whether it
  is good poetry is a human judgment.

## Tests

```
python -m unittest discover -s tests -v
```

The agent tests replace the model with scripted replies. They check the control flow: metre failures,
judge feedback, threshold stopping, and prose skipping the metre check.

## Layout

```
latinwriter/meter.py    scanner: elision, weights, hexameter and pentameter patterns
latinwriter/agent.py    generate / check / judge loop
latinwriter/prompts.py  form rules, style notes, generator, feedback and judge prompts
latinwriter/llm.py      Anthropic client, reply parsing
latinwriter/cli.py      check and write commands
tests/                  scanner and loop tests
```

## Corpus and evaluations

The corpus is collected from thelatinlibrary.com (`latinwriter/corpus.py`). It fetches 39 pages
(Vergil's Aeneid 1–12, Ovid's Metamorphoses 1–15, Caesar's Gallic War 1–8, four Cicero speeches) at one
request per second, with a descriptive User-Agent, caching each page in `data/raw/`. The site's
robots.txt is empty (no disallow rules). The texts are classical works. The corpus stays in `data/`
and is git-ignored; redistribute nothing from it except the scripts.

```
python -c "from latinwriter.corpus import build; build()"    # builds data/corpus.jsonl (about 24,800 records)
```

Split: sections with `section % 4 == 0` are the held-out test set; the rest is training. No
section is split between train and test.

### Evaluations (`evals/`)

| eval | command | what it measures |
|---|---|---|
| metre | `python -m evals.metre_eval` | the metre checker on held-out Aeneid/Metamorphoses lines (should pass) vs Caesar/Cicero sentences (should fail), swept over the violation tolerance |
| style | `python -m evals.style_eval` | author attribution from character n-grams, on held-out sections; saves the classifier to `data/style_model.pkl` |
| generation | `python -m evals.gen_eval` | the writer on 12 fixed cases, with and without the author exemplars: pass rate, metre pass rate, judge score, style match (needs `ANTHROPIC_API_KEY`) |

Results are written to `evals/results/` (git-ignored, since they quote corpus lines).

**Measured so far (offline):**

- Metre checker, default tolerance 1: recall 0.956 on held-out verse, false positives 4.2% on prose.
  Tolerance 0 gives recall 0.875 and 1.5% false positives. Tolerance 1 is the default.
- Author attribution: 74% accuracy across four authors (macro-F1 0.79). Caesar vs Cicero is 92%; Vergil
  vs Ovid is 73%.

**Not yet measured:** the generation eval. It needs an API key, and until it runs we don't know whether
the exemplars help.

### Exemplars

`latinwriter/agent.py` (`exemplar_block`) puts sample lines from the author's training split into the
generator's system prompt when the style is an author (vergil, ovid, caesar, cicero). It is on by default
and turned off with `--no-exemplars`. This is the "training" used here: no weights are changed, so
it is cheap to switch off and compare.

### Known limits

- The metre checker has no vowel-quantity dictionary. Its tolerance was chosen on this corpus, so the
  measured numbers are optimistic for other text.
- Macron-marked quantities in the Aeneid text are not used yet. They would give a per-syllable
  quantity eval, which is a natural next step.
- Fine-tuning a language model on this corpus was not done. It would need a GPU and a large training
  run, and the exemplar approach tests the same idea more cheaply first.
