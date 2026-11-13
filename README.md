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
