"""Per-ball match state and phase tagging."""

import polars as pl

from virtual_captain.config import load_format

INNINGS = ["match_id", "innings"]

# Ends of an innings that are not dismissals for scoring purposes.
NOT_OUT = ["retired hurt", "retired not out"]


def drop_super_overs(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.filter((pl.col("format") == "TEST") | (pl.col("innings") <= 2))


def add_ball_flags(lf: pl.LazyFrame) -> pl.LazyFrame:
    boundary = pl.col("non_boundary").not_()
    out = pl.col("wicket_type").is_not_null() & pl.col("wicket_type").is_in(NOT_OUT).not_()
    out2 = pl.col("other_wicket_type").is_not_null() & (
        pl.col("other_wicket_type").is_in(NOT_OUT).not_()
    )
    return lf.with_columns(
        (pl.col("runs_off_bat") + pl.col("extras")).alias("total_runs"),
        ((pl.col("runs_off_bat") == 4) & boundary).alias("is_four"),
        ((pl.col("runs_off_bat") == 6) & boundary).alias("is_six"),
        (out.cast(pl.Int8) + out2.cast(pl.Int8)).alias("wickets_on_ball"),
        (out & (pl.col("player_dismissed") == pl.col("striker"))).alias("striker_out"),
    )


def add_match_state(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Score, wickets and legal balls *before* each delivery, plus the chase target."""
    order = ["match_id", "innings", "delivery_number"]
    lf = lf.sort(order).with_columns(
        (pl.col("total_runs").cum_sum().over(INNINGS) - pl.col("total_runs")).alias("runs_before"),
        (pl.col("wickets_on_ball").cum_sum().over(INNINGS) - pl.col("wickets_on_ball")).alias(
            "wickets_before"
        ),
        (pl.col("is_legal").cast(pl.Int32).cum_sum().over(INNINGS) - pl.col("is_legal")).alias(
            "balls_before"
        ),
    )

    first_innings = (
        lf.filter(pl.col("innings") == 1)
        .group_by("match_id")
        .agg(pl.col("total_runs").sum().alias("first_innings_total"))
    )
    limited = pl.col("format") != "TEST"
    chasing = limited & (pl.col("innings") == 2)
    return (
        lf.join(first_innings, on="match_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(chasing).then(pl.col("first_innings_total") + 1).alias("target"),
        )
        .with_columns(
            (pl.col("target") - pl.col("runs_before")).alias("runs_required"),
        )
        .drop("first_innings_total")
    )


def _phase_expr(clock: pl.Expr, phases: list[dict]) -> pl.Expr:
    expr = None
    for p in reversed(phases):
        if p["until"] is None:
            expr = pl.lit(p["name"])
        else:
            expr = pl.when(clock < p["until"]).then(pl.lit(p["name"])).otherwise(expr)
    return expr


def add_phase(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Phase of play from configs/formats. Needs balls_before and balls_per_over."""
    t20, odi, test = load_format("t20"), load_format("odi"), load_format("test")
    five = t20["variants"]["five_ball"]
    is_five_ball = pl.col("balls_per_over") == 5
    balls = pl.col("balls_before")
    ball_age = pl.col("balls_before") % test["new_ball_every"]

    fmt = pl.col("format")
    phase = (
        pl.when((fmt == "T20") & is_five_ball)
        .then(_phase_expr(balls, five["phases"]))
        .when(fmt == "T20")
        .then(_phase_expr(balls, t20["phases"]))
        .when(fmt == "ODI")
        .then(_phase_expr(balls, odi["phases"]))
        .otherwise(_phase_expr(ball_age, test["phases"]))
    )
    scheduled = (
        pl.when((fmt == "T20") & is_five_ball)
        .then(five["scheduled_balls"])
        .when(fmt == "T20")
        .then(t20["scheduled_balls"])
        .when(fmt == "ODI")
        .then(odi["scheduled_balls"])
    )
    return lf.with_columns(
        phase.alias("phase"),
        ball_age.alias("ball_age"),
        (scheduled - pl.col("balls_before")).clip(lower_bound=0).alias("balls_remaining"),
    )


def add_era(lf: pl.LazyFrame) -> pl.LazyFrame:
    year = pl.col("start_date").dt.year()
    return lf.with_columns(
        year.alias("year"),
        pl.when(year <= 2013)
        .then(pl.lit("2001-13"))
        .when(year <= 2019)
        .then(pl.lit("2014-19"))
        .otherwise(pl.lit("2020+"))
        .alias("era"),
    )
