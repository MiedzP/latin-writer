import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GEN_MODEL = os.environ.get("LATINWRITER_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.environ.get("LATINWRITER_JUDGE_MODEL", GEN_MODEL)
