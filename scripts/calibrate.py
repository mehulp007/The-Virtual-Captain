"""Fit the fielding simulator to real scoring rates and check it on held-out seasons.

Needs the deliveries table and a trained shot model.

Usage:
  uv run python scripts/calibrate.py
  uv run python scripts/calibrate.py --trials 50 --refine 0 --per-group 10000   # quick run
"""

import argparse
import sys
import time

import polars as pl

from virtual_captain.models import encoding as enc
from virtual_captain.models.shot_model import ShotModel
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR, REPORTS_DIR
from virtual_captain.sim import calibrate as cal
from virtual_captain.sim.params import CALIBRATED

SHOT_MODEL = DATA_DIR / "models" / "shot_model.pt"


def compact(table: pl.DataFrame) -> pl.DataFrame:
    cols = ["runs_per_ball", *cal.RATES]
    return table.select(
        "format",
        "phase",
        "bowling",
        *[
            pl.format("{} / {}", pl.col(f"real_{c}").round(3), pl.col(f"sim_{c}").round(3)).alias(c)
            for c in cols
        ],
    )


def mean_abs_error(table: pl.DataFrame) -> dict[str, float]:
    w = table["balls"].to_numpy()
    out = {}
    for c in ["runs_per_ball", *cal.RATES]:
        rel = (table[f"sim_{c}"] / table[f"real_{c}"] - 1).abs().to_numpy()
        out[c] = round(float((rel * w).sum() / w.sum() * 100), 1)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trials", type=int, default=600, help="global TPE search")
    p.add_argument("--refine", type=int, default=400, help="CMA-ES refinement")
    p.add_argument("--per-group", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model = ShotModel.load(SHOT_MODEL)
    frame = (
        pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
        .pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .pipe(enc.split_column)
        .collect()
    )

    profiles = cal.deep_profiles(frame.filter(pl.col("split") == "train"))
    print(f"deep-scoring profiles for {len(profiles):,} batter-format pairs")

    def subset(split):
        rows = frame.filter(pl.col("split") == split)
        return cal.build_set(rows, model, args.per_group, profiles, args.seed)

    train, test = subset("train"), subset("test")
    print(f"calibrating on {train.rows.height:,} deliveries, checking on {test.rows.height:,}")

    t0 = time.perf_counter()
    params, best = cal.fit(train, n_trials=args.trials, n_refine=args.refine, seed=args.seed)
    took = time.perf_counter() - t0
    print(f"{args.trials} + {args.refine} trials in {took:.0f}s, best loss {best:.4f}")
    print({k: round(getattr(params, k), 3) for k in CALIBRATED})
    params.save(note="fitted by scripts/calibrate.py against seasons before 2025")

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3, tbl_hide_dataframe_shape=True):
        for name, data in [("train", train), ("held-out 2025-26", test)]:
            table = cal.report(data, params)
            print(f"\n{name}: loss {cal.loss(data, params):.4f}  (real / simulated)")
            print(compact(table))
            print("ball-weighted mean absolute error, %:", mean_abs_error(table))
            if data is test:
                table.write_csv(REPORTS_DIR / "sim_calibration.csv", float_precision=4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
