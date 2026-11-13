"""Text-only model client (Anthropic or Groq) and reply parsing.

Provider is chosen by LATINWRITER_PROVIDER (see config.py). Groq is reached through its
OpenAI-compatible endpoint, using the GROQ_API_KEY environment variable.
"""
import json
import os
import re

from .config import GEN_MODEL, GROQ_BASE_URL, PROVIDER

_client = None


def client():
    global _client
    if _client is None:
        if PROVIDER == "groq":
            import openai
            key = os.environ.get("GROQ_API_KEY")
            if not key:
                raise RuntimeError("Set GROQ_API_KEY (or use LATINWRITER_PROVIDER=anthropic).")
            _client = openai.OpenAI(api_key=key, base_url=GROQ_BASE_URL)
        else:
            import anthropic
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("Set ANTHROPIC_API_KEY (the `check` command needs no key).")
            _client = anthropic.Anthropic(api_key=key)
    return _client


def ask(messages: list[dict], system: str, model: str = GEN_MODEL, max_tokens: int = 4000) -> str:
    if PROVIDER == "groq":
        resp = client().chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}] + messages,
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return resp.choices[0].message.content or ""
    resp = client().messages.create(model=model, system=system, messages=messages, max_tokens=max_tokens)
    return "".join(b.text for b in resp.content if b.type == "text")


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    raw = fenced.group(1) if fenced else text[text.index("{"): text.rindex("}") + 1]
    return json.loads(raw)


def parse_output(reply: str) -> tuple[list[str], str]:
    """Split a reply of the form 'LATIN:\\n...\\nTRANSLATION:\\n...' into (latin_lines, translation)."""
    m = re.search(r"LATIN:\s*\n(.*?)(?:\nTRANSLATION:|\Z)", reply, re.S)
    latin = m.group(1) if m else reply
    t = re.search(r"TRANSLATION:\s*\n(.*)\Z", reply, re.S)
    translation = t.group(1).strip() if t else ""
    lines = [re.sub(r"^\s*\d+[.)]?\s*", "", ln).strip() for ln in latin.splitlines()]
    return [ln for ln in lines if ln], translation
