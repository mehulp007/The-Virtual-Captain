"""What a wicket is worth in runs, by format and phase.

A field is judged on runs conceded minus lambda x wickets taken, where lambda is how many runs
the batting side loses, on average, when a wicket falls at that stage:

  limited overs  expected runs to the end of the innings with w wickets down and b balls left,
                 minus the same with w + 1 down (first innings only, so chases that stop at the
                 target don't drag the curve down)
  Tests          expected runs to the end of the innings with w down, minus with w + 1 down

Only seasons before the hold-out year are used.
"""

import numpy as np
import polars as pl
from scipy.optimize import least_squares

from virtual_captain.models.encoding import HOLDOUT_FROM_YEAR

MIN_STATES = 30  # an (overs left, wickets) cell needs this many balls to be trusted


def _runs_to_end(lf: pl.LazyFrame) -> pl.LazyFrame:
    total = pl.col("total_runs").sum().over("match_id", "innings")
    return lf.with_columns((total - pl.col("runs_before")).alias("runs_to_end"))


def _base(deliveries: pl.LazyFrame) -> pl.LazyFrame:
    return deliveries.filter(
        (pl.col("gender") == "male")
        & pl.col("is_legal")
        & (pl.col("year") < HOLDOUT_FROM_YEAR)
        & (pl.col("wickets_before") <= 8)
    ).pipe(_runs_to_end)


def _limited(base: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """First-innings limited-overs balls, and the wicket value of each
    (format, overs left, wickets down) state."""
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
    return limited, drop


def _tests(base: pl.LazyFrame) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Test balls (innings 1-3; the fourth stops at the target) and the wicket value of each
    number of wickets down."""
    tests = base.filter((pl.col("format") == "TEST") & (pl.col("innings") <= 3))
    t_table = tests.group_by("wickets_before").agg(pl.col("runs_to_end").mean().alias("r"))
    t_next = t_table.select(
        (pl.col("wickets_before") - 1).alias("wickets_before"), pl.col("r").alias("r_next")
    )
    t_drop = t_table.join(t_next, on="wickets_before").select(
        "wickets_before", (pl.col("r") - pl.col("r_next")).alias("value")
    )
    return tests, t_drop


def wicket_values(deliveries: pl.LazyFrame) -> pl.DataFrame:
    """Columns: format, phase, wicket_value (runs), balls (deliveries it was averaged over)."""
    base = _base(deliveries)
    limited, drop = _limited(base)
    limited_values = (
        limited.join(drop, on=["format", "overs_left", "wickets_before"])
        .group_by("format", "phase")
        .agg(pl.col("value").mean().alias("wicket_value"), pl.len().alias("balls"))
    )
    tests, t_drop = _tests(base)
    test_values = (
        tests.join(t_drop, on="wickets_before")
        .group_by("format", "phase")
        .agg(pl.col("value").mean().alias("wicket_value"), pl.len().alias("balls"))
    )
    return pl.concat([limited_values, test_values]).sort("format", "phase").collect()


def _resource_curve(overs, wickets, z0, b, f):
    """Duckworth-Lewis form: expected runs still to come with `overs` left and `wickets` down."""
    fw = f[wickets]
    return z0 * fw * (1.0 - np.exp(-b * overs / fw))


def _fit_resources(table: pl.DataFrame) -> tuple[float, float, np.ndarray]:
    """Fit z0, b and F(0..9) (F(0) = 1, decreasing) to mean runs to the end of the innings in
    each (overs left, wickets down) state, weighting every state by how often it was seen."""
    o = table["overs_left"].to_numpy().astype(float)
    w = table["wickets_before"].to_numpy()
    r = table["r"].to_numpy()
    weight = np.sqrt(table["n"].to_numpy())

    def unpack(theta):
        z0, b = np.exp(theta[0]), np.exp(theta[1])
        f = np.concatenate([[1.0], np.cumprod(1.0 / (1.0 + np.exp(-theta[2:])))])
        return z0, b, f

    def residuals(theta):
        return weight * (_resource_curve(o, w, *unpack(theta)) - r)

    start = np.concatenate([[np.log(r.max()), np.log(r.max() / o.max())], np.full(9, 2.0)])
    fit = least_squares(residuals, start)
    return unpack(fit.x)


def wicket_value_states(deliveries: pl.LazyFrame) -> pl.DataFrame:
    """What a wicket is worth in each match state, rather than averaged over a phase.

    Limited overs: a Duckworth-Lewis resource curve is fitted to expected runs to the end of
    the first innings, by overs left and wickets down, and a wicket is worth the drop in that
    curve when one more is down (the tenth is worth everything still to come). The fit smooths
    out sparse states. Tests: expected runs to the end of the innings by wickets down, as for
    the phase values.

    Columns: format, overs_left (null for Tests), wickets_before (0-9), wicket_value.
    """
    # runs to the end counted over every delivery (wides and no-balls included), then states
    # taken at legal balls
    total = pl.col("total_runs").sum().over("match_id", "innings")
    base = (
        deliveries.filter((pl.col("gender") == "male") & (pl.col("year") < HOLDOUT_FROM_YEAR))
        .with_columns((total - pl.col("runs_before")).alias("runs_to_end"))
        .filter(pl.col("is_legal"))
    )
    frames = []
    overs_in = {"T20": 20, "ODI": 50}
    states = (
        base.filter((pl.col("format") != "TEST") & (pl.col("innings") == 1))
        .with_columns(((pl.col("balls_remaining") + 5) // 6).alias("overs_left"))
        .filter(pl.col("overs_left") >= 1)
        .group_by("format", "overs_left", "wickets_before")
        .agg(pl.col("runs_to_end").mean().alias("r"), pl.len().alias("n"))
        .filter(pl.col("n") >= MIN_STATES)
        .collect()
    )
    for fmt, total in overs_in.items():
        z0, b, f = _fit_resources(states.filter(pl.col("format") == fmt))
        o = np.repeat(np.arange(1, total + 1), 10).astype(float)
        w = np.tile(np.arange(10), total)
        now = _resource_curve(o, w, z0, b, f)
        after = np.where(w < 9, _resource_curve(o, np.minimum(w + 1, 9), z0, b, f), 0.0)
        frames.append(
            pl.DataFrame(
                {
                    "format": fmt,
                    "overs_left": o.astype(np.int64),
                    "wickets_before": w.astype(np.int64),
                    "wicket_value": now - after,
                }
            )
        )

    tests = (
        base.filter((pl.col("format") == "TEST") & (pl.col("innings") <= 3))
        .group_by("wickets_before")
        .agg(pl.col("runs_to_end").mean().alias("r"))
        .sort("wickets_before")
        .collect()
    )
    r = dict(zip(tests["wickets_before"], tests["r"], strict=True))
    frames.append(
        pl.DataFrame(
            {
                "format": "TEST",
                "overs_left": pl.Series([None] * 10, dtype=pl.Int64),
                "wickets_before": np.arange(10, dtype=np.int64),
                "wicket_value": [r[w] - r.get(w + 1, 0.0) for w in range(10)],
            }
        )
    )
    return pl.concat(frames)


def as_dict(values: pl.DataFrame) -> dict[tuple[str, str], float]:
    rows = values.select("format", "phase", "wicket_value").iter_rows()
    return {(f, p): float(v) for f, p, v in rows}
