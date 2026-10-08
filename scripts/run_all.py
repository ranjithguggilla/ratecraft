"""Rebuild every number, table and chart in the README from the raw data.

    python scripts/run_all.py                    # everything, a little over two minutes on four cores
    python scripts/run_all.py --stage severity   # one stage, reusing cached earlier stages
    python scripts/run_all.py --synthetic 20000  # smoke test on a synthetic book, no download

Stages cache their fitted models under data/interim, and the final ``report``
stage writes results/, assets/ and the shipped pricing model.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ratecraft.config import SEED, Paths
from ratecraft.data import load
from ratecraft.pipeline import STAGES, prepare, run_stage
from ratecraft.report import build_report
from ratecraft.synth import make_portfolio

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", action="append", choices=[*STAGES, "report"],
                        help="run only this stage (repeatable); default runs all of them")
    parser.add_argument("--synthetic", type=int, default=0, metavar="N",
                        help="use a synthetic book of N policies instead of the real data")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root to read from and write to")
    args = parser.parse_args(argv)

    started = time.time()
    paths = Paths(args.root).ensure()
    if args.synthetic:
        policies, claims = make_portfolio(args.synthetic, seed=SEED)
        interim = paths.interim / "synthetic"
    else:
        policies, claims = load(paths.raw, paths.interim)
        interim = paths.interim / "real"
    print(f"{len(policies):,} policies, {len(claims):,} claims")
    ctx = prepare(policies, claims, interim, quick=bool(args.synthetic))
    print("split:", ctx.split.sizes())

    for stage in args.stage or [*STAGES, "report"]:
        if stage == "report":
            build_report(ctx, paths, synthetic=bool(args.synthetic))
        else:
            run_stage(ctx, stage)
    print(f"done in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
