import argparse
import sys
from pathlib import Path

from .agent import write
from .meter import scan, scan_couplet
from .prompts import FORMS, STYLES


def _print_scan(results) -> bool:
    all_ok = True
    for i, r in enumerate(results, 1):
        status = "ok  " if r.ok else "FAIL"
        all_ok &= r.ok
        pattern = f"  {r.pattern}" if r.pattern else ""
        print(f"{i:>2} {status} [{r.syllables:>2} syl]{pattern}  {r.line}")
        if not r.ok:
            print(f"        {r.reason}")
    return all_ok


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="latin-writer", description="write and check Latin verse and prose")
    sub = ap.add_subparsers(dest="cmd", required=True)

    chk = sub.add_parser("check", help="metre-check a file of Latin lines (no API key needed)")
    chk.add_argument("file", type=Path)
    chk.add_argument("--form", choices=sorted(FORMS), default="hexameter")

    wr = sub.add_parser("write", help="generate a piece with the agent loop (needs ANTHROPIC_API_KEY)")
    wr.add_argument("--form", choices=sorted(FORMS), required=True)
    wr.add_argument("--style", choices=sorted(STYLES), default="plain")
    wr.add_argument("--theme", required=True, help="what the piece is about, in English")
    wr.add_argument("--lines", type=int, default=6)
    wr.add_argument("--out", type=Path, required=True)
    wr.add_argument("--iters", type=int, default=4)
    wr.add_argument("--threshold", type=int, default=80)

    args = ap.parse_args(argv)
    if args.cmd == "check":
        lines = [ln for ln in args.file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        results = scan_couplet(lines) if args.form == "elegiac" else [scan(ln, args.form) for ln in lines]
        sys.exit(0 if _print_scan(results) else 1)

    if args.cmd == "write":
        report = write(
            form=args.form, style=args.style, theme=args.theme, out=args.out,
            lines=args.lines, max_iters=args.iters, threshold=args.threshold,
        )
        print(f"result: {report['result']}  (piece: {args.out / 'piece.txt'})")


if __name__ == "__main__":
    main()
