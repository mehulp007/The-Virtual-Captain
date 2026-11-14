"""What a wicket is worth in runs, by format and phase.

A field is judged on runs conceded minus lambda x wickets taken, where lambda is how many runs
the batting side loses, on average, when a wicket falls at that stage:

  limited overs  expected runs to the end of the innings with w wickets down and b balls left,
                 minus the same with w + 1 down (first innings only, so chases that stop at the
                 target don't drag the curve down)
  Tests          expected runs to the end of the innings with w down, minus with w + 1 down

Only seasons before the hold-out year are used.
"""

import polars as pl

from virtual_captain.models.encoding import HOLDOUT_FROM_YEAR

MIN_STATES = 30  # an (overs left, wickets) cell needs this many balls to be trusted


def _runs_to_end(lf: pl.LazyFrame) -> pl.LazyFrame:
    total = pl.col("total_runs").sum().over("match_id", "innings")
    return lf.with_columns((total - pl.col("runs_before")).alias("runs_to_end"))


def wicket_values(deliveries: pl.LazyFrame) -> pl.DataFrame:
    """Columns: format, phase, wicket_value (runs), balls (deliveries it was averaged over)."""
    base = deliveries.filter(
        (pl.col("gender") == "male")
        & pl.col("is_legal")
        & (pl.col("year") < HOLDOUT_FROM_YEAR)
        & (pl.col("wickets_before") <= 8)
    ).pipe(_runs_to_end)

    # limited overs: state = (format, overs left, wickets down)
    limited = base.filter((pl.col("format") != "TEST") & (pl.col("innings") == 1)).with_columns(
        ((pl.col("balls_remaining") + 5) // 6).alias("overs_left")
    )
    keys = ["format", "overs_left"]
    table = (
        limited.group_by(*keys, "wickets_before")
        .agg(pl.col("runs_to_end").mean().alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_STATES)
    )
    next_w = table.select(
        *keys, (pl.col("wickets_before") - 1).alias("wickets_before"), pl.col("r").alias("r_next")
    )
    drop = table.join(next_w, on=[*keys, "wickets_before"]).select(
        *keys, "wickets_before", (pl.col("r") - pl.col("r_next")).alias("value")
    )
    limited_values = (
        limited.join(drop, on=[*keys, "wickets_before"])
        .group_by("format", "phase")
        .agg(pl.col("value").mean().alias("wicket_value"), pl.len().alias("balls"))
    )

    # Tests: state = wickets down (innings 1-3; the fourth stops at the target)
    tests = base.filter((pl.col("format") == "TEST") & (pl.col("innings") <= 3))
    t_table = tests.group_by("wickets_before").agg(pl.col("runs_to_end").mean().alias("r"))
    t_next = t_table.select(
        (pl.col("wickets_before") - 1).alias("wickets_before"), pl.col("r").alias("r_next")
    )
    t_drop = t_table.join(t_next, on="wickets_before").select(
        "wickets_before", (pl.col("r") - pl.col("r_next")).alias("value")
    )
    test_values = (
        tests.join(t_drop, on="wickets_before")
        .group_by("format", "phase")
        .agg(pl.col("value").mean().alias("wicket_value"), pl.len().alias("balls"))
    )
    return pl.concat([limited_values, test_values]).sort("format", "phase").collect()


def as_dict(values: pl.DataFrame) -> dict[tuple[str, str], float]:
    rows = values.select("format", "phase", "wicket_value").iter_rows()
    return {(f, p): float(v) for f, p, v in rows}
