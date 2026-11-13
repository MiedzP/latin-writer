import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Provider for the writer and the judge: "anthropic" (default) or "groq" (free tier, OpenAI-compatible).
PROVIDER = os.environ.get("LATINWRITER_PROVIDER", "anthropic").lower()
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

if PROVIDER == "groq":
    GEN_MODEL = os.environ.get("LATINWRITER_MODEL", "openai/gpt-oss-120b")
    JUDGE_MODEL = os.environ.get("LATINWRITER_JUDGE_MODEL", GEN_MODEL)
else:
    GEN_MODEL = os.environ.get("LATINWRITER_MODEL", "claude-sonnet-5-5")
    JUDGE_MODEL = os.environ.get("LATINWRITER_JUDGE_MODEL", GEN_MODEL)
