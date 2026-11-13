"""Lazy readers for the raw CSVs. Everything is read as text and cast explicitly, which is
faster and safer than letting the CSV reader guess types over 5.8M rows."""

from pathlib import Path

import polars as pl

from virtual_captain.paths import RAW_DIR

ULTIMATE_DIR = RAW_DIR / "ultimate"

KEYS = ["match_id", "innings", "delivery_number"]

_BOOL = {"True": True, "False": False}


def _bool(name: str) -> pl.Expr:
    return pl.col(name).replace_strict(_BOOL, default=None, return_dtype=pl.Boolean)


def scan_balls(raw_dir: Path = ULTIMATE_DIR) -> pl.LazyFrame:
    lf = pl.scan_csv(raw_dir / "ball_by_ball_data.csv", infer_schema_length=0)
    ints = ["match_id", "innings", "delivery_number", "over", "ball_in_over", "runs_off_bat"]
    ints += ["extras", "wides", "noballs", "byes", "legbyes", "penalty"]
    return lf.select(
        *[pl.col(c).cast(pl.Int32) for c in ints],
        pl.col("start_date").str.to_date(),
        "season",
        _bool("is_legal_delivery").alias("is_legal"),
        "batting_team",
        "bowling_team",
        "striker",
        "striker_id",
        "non_striker",
        "bowler",
        "bowler_id",
        "wicket_type",
        "player_dismissed",
        "other_wicket_type",
        pl.col("striker balls faced").cast(pl.Int32).alias("striker_balls_faced"),
        pl.col("total striker runs").cast(pl.Int32).alias("striker_runs"),
        "gender",
        "event",
        "format",
        pl.col("type").alias("competition_type"),
        pl.col("batting style_striker").alias("batting_style"),
        pl.col("bowling style_bowler").alias("bowling_style"),
        "fielder_1",
        "fielder_1_id",
        _bool("non_boundary"),
    )


def scan_tracking(raw_dir: Path = ULTIMATE_DIR) -> pl.LazyFrame:
    lf = pl.scan_csv(raw_dir / "cricinfo_tracking.csv", infer_schema_length=0)
    return lf.select(
        *[pl.col(c).cast(pl.Int32) for c in KEYS],
        "ball_line",
        "ball_length",
        "shot_played",
        "shot_control",
        pl.col("wagon_zone").cast(pl.Int8),
        pl.col("wagon_angle").cast(pl.Float32),
        pl.col("wagon_distance").cast(pl.Float32),
        _bool("wagon_in_ring"),
        "shot_direction",
        "ball_events",
    )


def scan_matches(raw_dir: Path = ULTIMATE_DIR) -> pl.LazyFrame:
    lf = pl.scan_csv(raw_dir / "matches.csv", infer_schema_length=0)
    return lf.select(
        pl.col("match_id").cast(pl.Int32),
        "venue",
        "city",
        pl.col("balls_per_over").cast(pl.Int8),
    )
