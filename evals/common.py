"""Shared helpers for the evaluations."""
import json
from pathlib import Path

from latinwriter.config import ROOT
from latinwriter.corpus import load

RESULTS = ROOT / "evals" / "results"


def clean(text: str) -> str:
    """Remove editorial brackets that mark disputed lines (they are still real verse/prose)."""
    return text.replace("[", "").replace("]", "").strip()


def save(name: str, payload: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / name
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def held_out(kind: str) -> list[dict]:
    return [dict(r, text=clean(r["text"])) for r in load(split="test", kind=kind)]


def train(kind: str) -> list[dict]:
    return [dict(r, text=clean(r["text"])) for r in load(split="train", kind=kind)]
