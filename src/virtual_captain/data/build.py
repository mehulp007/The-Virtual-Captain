"""Join the raw ball-by-ball, tracking and match files into one deliveries table."""

from pathlib import Path

import polars as pl

from virtual_captain.data import features, loaders
from virtual_captain.data.coords import wagon_xy_exprs
from virtual_captain.field.geometry import PITCH_HALF
from virtual_captain.models import context

# First season held out for validation; venue features are computed from earlier years.
HOLDOUT_FROM_YEAR = 2025

COLUMNS = [
    # identity
    "match_id", "innings", "delivery_number", "over", "ball_in_over", "start_date", "year",
    "season", "era", "gender", "format", "competition_type", "event", "venue", "venue_key",
    "batting_team", "bowling_team",
    # players
    "striker", "striker_id", "non_striker", "batting_style", "left_handed",
    "bowler", "bowler_id", "bowling_style", "bowler_arm", "bowler_kind", "pace_or_spin",
    "spin_turn", "arm_vs_bat",
    # state before the ball
    "phase", "balls_before", "ball_age", "balls_remaining", "runs_before", "wickets_before",
    "target", "runs_required", "striker_balls_faced", "striker_runs",
    # outcome
    "is_legal", "runs_off_bat", "extras", "total_runs", "wides", "noballs", "byes", "legbyes",
    "is_four", "is_six", "non_boundary", "wicket_type", "player_dismissed", "striker_out",
    "wickets_on_ball", "fielder_1", "fielder_1_id",
    # tracking
    "ball_line", "ball_length", "shot_played", "shot_control", "has_wagon", "wagon_zone",
    "wagon_angle", "wagon_distance", "wagon_in_ring", "shot_direction", "x_m", "y_m",
    "shot_length_m", "ball_events",
    # venue
    "venue_balls", "venue_runs_per_ball", "venue_boundary_rate", "venue_spin_wicket_share",
]  # fmt: skip


def build_deliveries(raw_dir: Path = loaders.ULTIMATE_DIR) -> pl.DataFrame:
    balls = loaders.scan_balls(raw_dir)
    styles = balls.select("bowling_style").unique().collect()["bowling_style"]

    lf = (
        balls.pipe(features.drop_super_overs)
        .pipe(features.add_ball_flags)
        .pipe(features.add_match_state)
        .join(loaders.scan_matches(raw_dir), on="match_id", how="left")
        .pipe(features.add_phase)
        .pipe(features.add_era)
        .join(loaders.scan_tracking(raw_dir), on=loaders.KEYS, how="left")
        .with_columns(*wagon_xy_exprs())
        .with_columns(
            pl.col("wagon_angle").is_not_null().alias("has_wagon"),
            # distance the ball travelled from the striker's end, not from the pitch centre
            (pl.col("x_m").pow(2) + (pl.col("y_m") + PITCH_HALF).pow(2))
            .sqrt()
            .alias("shot_length_m"),
            context.venue_key(pl.col("venue")).alias("venue_key"),
        )
        .pipe(context.add_bowler_context, styles)
    )

    df = lf.collect()
    venues = context.venue_features(df.lazy(), before_year=HOLDOUT_FROM_YEAR)
    df = df.join(venues, on=["venue_key", "format", "gender"], how="left")
    return df.select(COLUMNS).sort(loaders.KEYS)


def coverage_table(df: pl.DataFrame) -> pl.DataFrame:
    """Share of legal deliveries carrying each tracking field, by gender, format and era."""
    legal = df.filter(pl.col("is_legal"))
    return (
        legal.group_by("gender", "format", "era")
        .agg(
            pl.len().alias("balls"),
            pl.col("has_wagon").mean().alias("wagon"),
            pl.col("ball_line").is_not_null().mean().alias("line"),
            pl.col("ball_length").is_not_null().mean().alias("length"),
            pl.col("shot_played").is_not_null().mean().alias("shot"),
            pl.col("shot_control").is_not_null().mean().alias("control"),
        )
        .sort("gender", "format", "era", descending=[True, False, False])
    )
