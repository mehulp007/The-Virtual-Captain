"""Bowler type, how it relates to the batter, and venue ("pitch") features."""

import polars as pl

# Cricinfo bowling styles can list several; the first is the bowler's main style.
WRIST_SPIN = ("legbreak", "googly", "wrist-spin")
FINGER_SPIN = ("offbreak", "orthodox")


def parse_bowling_style(style: str | None) -> dict:
    """Split a Cricinfo bowling style into arm, kind and pace/spin.

    kind is one of: fast, medium, finger_spin, wrist_spin, other_spin, unknown.
    """
    if not style:
        return {"bowler_arm": None, "bowler_kind": "unknown", "pace_or_spin": None}
    s = style.split(",")[0].strip().lower()

    if "left-arm" in s or "slow left-arm" in s:
        arm = "left"
    elif "right-arm" in s or s.startswith(("legbreak", "offbreak")):
        arm = "right"
    else:
        arm = None

    if any(k in s for k in WRIST_SPIN):
        kind = "wrist_spin"
    elif any(k in s for k in FINGER_SPIN):
        kind = "finger_spin"
    elif "fast" in s:
        kind = "fast"
    elif "medium" in s:
        kind = "medium"
    elif "slow" in s:
        kind = "other_spin"
    else:
        kind = "unknown"

    if kind in ("fast", "medium"):
        pace_or_spin = "pace"
    elif kind == "unknown":
        pace_or_spin = None
    else:
        pace_or_spin = "spin"
    return {"bowler_arm": arm, "bowler_kind": kind, "pace_or_spin": pace_or_spin}


def bowler_table(styles: pl.Series) -> pl.DataFrame:
    """Lookup table from raw bowling style to parsed columns."""
    rows = [{"bowling_style": s, **parse_bowling_style(s)} for s in styles.unique().to_list()]
    return pl.DataFrame(
        rows,
        schema={
            "bowling_style": pl.String,
            "bowler_arm": pl.String,
            "bowler_kind": pl.String,
            "pace_or_spin": pl.String,
        },
    )


def add_bowler_context(lf: pl.LazyFrame, styles: pl.Series) -> pl.LazyFrame:
    lf = lf.join(bowler_table(styles).lazy(), on="bowling_style", how="left")
    lhb = pl.col("batting_style") == "left-hand bat"
    # For a right-hander, off-spin and left-arm wrist spin turn in; leg-spin and left-arm
    # orthodox turn away. A left-hander flips that.
    rhb_into = (pl.col("bowler_arm") == "right") == (pl.col("bowler_kind") == "finger_spin")
    is_turning = pl.col("bowler_kind").is_in(["finger_spin", "wrist_spin"])
    return lf.with_columns(
        pl.when(pl.col("batting_style").is_null()).then(None).otherwise(lhb).alias("left_handed"),
        pl.when(is_turning & pl.col("batting_style").is_not_null())
        .then(pl.when(rhb_into != lhb).then(pl.lit("in")).otherwise(pl.lit("away")))
        .alias("spin_turn"),
        # right-arm bowler to right-hander = same side; left-arm to right-hander = opposite
        pl.when(pl.col("bowler_arm").is_not_null() & pl.col("batting_style").is_not_null())
        .then(
            pl.when((pl.col("bowler_arm") == "left") == lhb)
            .then(pl.lit("same"))
            .otherwise(pl.lit("opposite"))
        )
        .alias("arm_vs_bat"),
    )


def venue_key(venue: pl.Expr) -> pl.Expr:
    """'Wankhede Stadium, Mumbai' and 'Wankhede Stadium' should be the same ground."""
    return venue.str.split(",").list.first().str.strip_chars().str.to_lowercase()


def venue_features(
    deliveries: pl.LazyFrame,
    before_year: int,
    prior_balls: float = 3000.0,
    prior_wickets: float = 60.0,
) -> pl.DataFrame:
    """Per (venue, format, gender) scoring and spin-friendliness, shrunk to the format mean.

    Only matches before `before_year` are used so held-out seasons don't leak into the
    features they are evaluated on.
    """
    legal = deliveries.filter(pl.col("is_legal") & (pl.col("year") < before_year))
    bowler_wkt = pl.col("striker_out") & pl.col("wicket_type").is_in(["run out"]).not_()
    per_venue = legal.group_by("venue_key", "format", "gender").agg(
        pl.len().alias("venue_balls"),
        pl.col("total_runs").sum().alias("_runs"),
        (pl.col("is_four") | pl.col("is_six")).sum().alias("_boundaries"),
        bowler_wkt.sum().alias("_wkts"),
        (bowler_wkt & (pl.col("pace_or_spin") == "spin")).sum().alias("_spin_wkts"),
    )
    base = per_venue.group_by("format", "gender").agg(
        (pl.col("_runs").sum() / pl.col("venue_balls").sum()).alias("_rpb0"),
        (pl.col("_boundaries").sum() / pl.col("venue_balls").sum()).alias("_bnd0"),
        (pl.col("_spin_wkts").sum() / pl.col("_wkts").sum()).alias("_spin0"),
    )
    k, kw = prior_balls, prior_wickets
    return (
        per_venue.join(base, on=["format", "gender"])
        .with_columns(
            ((pl.col("_runs") + k * pl.col("_rpb0")) / (pl.col("venue_balls") + k)).alias(
                "venue_runs_per_ball"
            ),
            ((pl.col("_boundaries") + k * pl.col("_bnd0")) / (pl.col("venue_balls") + k)).alias(
                "venue_boundary_rate"
            ),
            ((pl.col("_spin_wkts") + kw * pl.col("_spin0")) / (pl.col("_wkts") + kw)).alias(
                "venue_spin_wicket_share"
            ),
        )
        .select(
            "venue_key",
            "format",
            "gender",
            "venue_balls",
            "venue_runs_per_ball",
            "venue_boundary_rate",
            "venue_spin_wicket_share",
        )
        .collect()
    )
