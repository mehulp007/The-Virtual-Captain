"""Build data/processed/deliveries.parquet from the raw CSVs and write a coverage report.

Usage:
  uv run python scripts/build_dataset.py
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl

from virtual_captain.data.build import build_deliveries, coverage_table
from virtual_captain.data.loaders import ULTIMATE_DIR
from virtual_captain.paths import PROCESSED_DIR, REPORTS_DIR


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=ULTIMATE_DIR)
    p.add_argument("--out", type=Path, default=PROCESSED_DIR / "deliveries.parquet")
    args = p.parse_args(argv)

    if not (args.raw / "ball_by_ball_data.csv").exists():
        print(f"no raw data in {args.raw}, run scripts/download_data.py first")
        return 1

    t0 = time.perf_counter()
    df = build_deliveries(args.raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(args.out, compression="zstd")
    size_mb = args.out.stat().st_size / 1e6
    print(f"wrote {df.height:,} deliveries to {args.out} ({size_mb:.0f} MB)")
    print(f"took {time.perf_counter() - t0:.0f}s")

    cov = coverage_table(df)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cov.write_csv(REPORTS_DIR / "coverage.csv", float_precision=3)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        print(cov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
