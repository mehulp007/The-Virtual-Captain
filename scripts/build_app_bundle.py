"""Pack what the demo app needs into data/app/bundle.npz (see virtual_captain.app.bundle).

Needs the captain's data (scripts/build_captain_data.py), a trained captain and the deliveries
table, for the real wagon wheels. The app itself needs none of them.

Usage:
  uv run python scripts/build_app_bundle.py
  uv run python scripts/build_app_bundle.py --captain data/models/captain.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from stable_baselines3 import PPO

from virtual_captain.app.bundle import BUNDLE, WAGON, from_captain_data, policy_weights
from virtual_captain.models import encoding as enc
from virtual_captain.paths import DATA_DIR, PROCESSED_DIR
from virtual_captain.rl.data import CaptainData

CAPTAIN = DATA_DIR / "models" / "captain_selfplay.zip"


def real_shots(contexts: pl.DataFrame) -> tuple[dict, np.ndarray, np.ndarray]:
    """For every context: up to WAGON real shots kept in the ground (theta, dist, runs, with
    -1 for a catch), the share of balls hit for six, and whether the batter bats left-handed.
    Same deliveries the captain's data was built from: tracked innings before 2025."""
    frame = (
        pl.scan_parquet(PROCESSED_DIR / "deliveries.parquet")
        .pipe(enc.tracked_innings)
        .pipe(enc.add_labels)
        .pipe(enc.split_column)
        .filter(pl.col("split") != "test")
        .with_columns(
            pl.when(pl.col("pace_or_spin") == "spin")
            .then(pl.lit("spin"))
            .otherwise(pl.lit("pace"))
            .alias("bowling"),
            pl.when(
                pl.col("striker_out") & pl.col("wicket_type").is_in(["caught", "caught and bowled"])
            )
            .then(-1)
            .otherwise(pl.col("runs_off_bat"))
            .alias("runs"),
        )
        .select(
            "striker_id",
            "format",
            "phase",
            "bowling",
            "outcome",
            "theta",
            "dist",
            "runs",
            "left_handed",
        )  # fmt: skip
        .collect()
    )
    shots, six, left = {}, np.zeros(contexts.height), np.zeros(contexts.height, bool)
    for c, who, fmt, phase, bowling in contexts.select(
        "context", "batter_id", "format", "phase", "bowling"
    ).iter_rows():
        cond = (
            (pl.col("format") == fmt) & (pl.col("phase") == phase) & (pl.col("bowling") == bowling)
        )
        if who is not None:
            cond &= pl.col("striker_id") == who
        rows = frame.filter(cond)
        six[c] = float((rows["outcome"] == enc.SIX).mean())
        if who is not None:
            left[c] = bool(rows["left_handed"].mean() > 0.5)
        kept = rows.filter(pl.col("outcome") == enc.IN_PLAY)
        kept = kept.sample(min(WAGON, kept.height), seed=c)
        shots[c] = (
            kept["theta"].to_numpy(),
            kept["dist"].to_numpy(),
            kept["runs"].to_numpy(),
        )
    return shots, six, left


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--captain", type=Path, default=CAPTAIN)
    p.add_argument("--out", type=Path, default=BUNDLE)
    args = p.parse_args(argv)

    data = CaptainData.load()
    model = PPO.load(
        args.captain,
        device="cpu",
        custom_objects={"learning_rate": 0.0, "lr_schedule": lambda _: 0.0},
    )
    shots, six, left = real_shots(data.contexts)
    bundle = from_captain_data(
        data, policy_weights(model), wagon=shots, real_six=six, left_handed=left
    )
    bundle.save(args.out)
    size = args.out.stat().st_size / 1e6
    print(f"{len(bundle.batter)} contexts, {len(bundle.wagon_runs)} real shots -> {args.out} "
          f"({size:.0f} MB)")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
