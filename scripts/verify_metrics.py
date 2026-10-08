"""Diff a fresh metrics.json against the committed one.

    python scripts/verify_metrics.py committed.json fresh.json

CI keeps a copy of the committed file, reruns the whole study from the raw
data, and calls this. Every number has to come back within 1% (or 1e-4 in
absolute terms for values near zero); strings, flags and counts have to match
exactly, so the data audit, the split sizes, the chosen interactions and the
shipped model are all pinned.

Skipped on purpose: timings, the booster's internal round count and tree
shape, and the worded reasons on the demo quotes. LightGBM's floating-point
sums can differ in the last bits between CPU architectures, and one different
split cascades into a round count that is off by one or a percentage that
rounds the other way. Those are not results; the scores built on them are,
and they stay checked. On the same machine a rerun reproduces every digit.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

RTOL = 1e-2
ATOL = 1e-4
SKIP = {"stage_seconds", "seconds", "frequency_gbm", "reasons"}


def _close(a: float, b: float) -> bool:
    if math.isnan(a) or math.isnan(b):
        # A standard deviation over a single seed is NaN by definition.
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(ATOL, RTOL * max(abs(a), abs(b)))


def compare(committed, fresh, path: str = "") -> list[str]:
    """Return a list of human-readable differences; empty means it reproduces."""
    problems: list[str] = []
    if isinstance(committed, dict) and isinstance(fresh, dict):
        for key in committed:
            if key in SKIP:
                continue
            where = f"{path}.{key}" if path else key
            if key not in fresh:
                problems.append(f"{where}: missing from the fresh run")
                continue
            problems += compare(committed[key], fresh[key], where)
        return problems
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return [f"{path}: {len(committed)} entries committed, {len(fresh)} now"]
        for i, (a, b) in enumerate(zip(committed, fresh)):
            problems += compare(a, b, f"{path}[{i}]")
        return problems
    numeric = (int, float)
    if isinstance(committed, bool) or isinstance(fresh, bool) or not (
        isinstance(committed, numeric) and isinstance(fresh, numeric)
    ):
        return [] if committed == fresh else [f"{path}: {committed!r} committed, {fresh!r} now"]
    if isinstance(committed, int) and isinstance(fresh, int):
        return [] if committed == fresh else [f"{path}: {committed} committed, {fresh} now"]
    return [] if _close(float(committed), float(fresh)) else [f"{path}: {committed} committed, {fresh} now"]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2:
        print(__doc__)
        return 2
    committed = json.loads(Path(args[0]).read_text())
    fresh = json.loads(Path(args[1]).read_text())
    problems = compare(committed, fresh)
    if problems:
        print(f"{len(problems)} number(s) moved:")
        for line in problems[:50]:
            print("  " + line)
        return 1
    print("every committed number reproduces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
