"""Verify the shipped artefacts against checksums.sha256.

Portable and line-ending tolerant: unlike `sha256sum -c`, it does not care whether a checkout
converted the list to CRLF, which is why the install scripts call it before touching anything.

    python3 scripts/verify-checksums.py            # every entry
    python3 scripts/verify-checksums.py --quiet    # only failures (used by the install scripts)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _probe import load_checksums, repo_root, verify_artefact  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the shipped files against checksums.sha256.")
    parser.add_argument("--quiet", action="store_true", help="print only failures")
    args = parser.parse_args()

    root = repo_root()
    entries = load_checksums(root)
    if not entries:
        raise SystemExit(f"checksums.sha256 is missing or empty in {root}")

    failures = 0
    for relative in entries:
        ok, detail = verify_artefact(root, relative, entries)
        if not ok:
            failures += 1
            print(f"FAIL {detail}")
        elif not args.quiet:
            print(f"OK   {detail}")
    if failures:
        raise SystemExit(f"{failures} of {len(entries)} shipped file(s) do not match checksums.sha256")
    if not args.quiet:
        print(f"\nall {len(entries)} shipped files match checksums.sha256")


if __name__ == "__main__":
    main()